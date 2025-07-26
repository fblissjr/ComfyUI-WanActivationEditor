"""
DuckDB-based storage for embeddings and vector operations.
Handles compression, deduplication, and memory-efficient operations.
"""

import os
import json
import hashlib
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import duckdb
import numpy as np
import torch
import zstandard as zstd


class EmbeddingDatabase:
    """Manages embedding storage and vector operations using DuckDB."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection and create schema."""
        if db_path is None:
            # Default to user's ComfyUI models directory
            base_path = Path.home() / "ComfyUI" / "models" / "wan_embeddings"
            base_path.mkdir(parents=True, exist_ok=True)
            db_path = str(base_path / "embeddings.duckdb")
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.compressor = zstd.ZstdCompressor(level=3)
        self.decompressor = zstd.ZstdDecompressor()
        
        self._init_schema()
    
    def _init_schema(self):
        """Create database schema if not exists."""
        schema_sql = """
        -- Core embedding storage
        CREATE TABLE IF NOT EXISTS embeddings (
            id VARCHAR PRIMARY KEY,
            prompt TEXT NOT NULL,
            prompt_hash VARCHAR NOT NULL UNIQUE,
            embedding_compressed BLOB NOT NULL,
            embedding_shape JSON NOT NULL,
            embedding_dtype VARCHAR NOT NULL,
            compression_ratio FLOAT,
            model_version VARCHAR NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1
        );
        
        -- Vector operations tracking
        CREATE TABLE IF NOT EXISTS vector_operations (
            id VARCHAR PRIMARY KEY,
            operation_type VARCHAR NOT NULL,
            result_embedding_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER,
            memory_peak_mb FLOAT
        );
        
        -- Operation inputs
        CREATE TABLE IF NOT EXISTS operation_inputs (
            operation_id VARCHAR,
            embedding_id VARCHAR,
            input_role VARCHAR NOT NULL,
            weight FLOAT DEFAULT 1.0,
            PRIMARY KEY (operation_id, embedding_id, input_role)
        );
        
        -- Operation parameters
        CREATE TABLE IF NOT EXISTS operation_parameters (
            operation_id VARCHAR,
            parameter_name VARCHAR NOT NULL,
            parameter_value JSON NOT NULL,
            PRIMARY KEY (operation_id, parameter_name)
        );
        
        -- Indices
        CREATE INDEX IF NOT EXISTS idx_prompt_hash ON embeddings(prompt_hash);
        CREATE INDEX IF NOT EXISTS idx_embedding_access ON embeddings(last_accessed DESC);
        CREATE INDEX IF NOT EXISTS idx_operation_type ON vector_operations(operation_type);
        """
        
        self.conn.execute(schema_sql)
    
    @contextmanager
    def _memory_context(self):
        """Context manager for VRAM-safe operations."""
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            torch.cuda.empty_cache()
    
    def _compress_array(self, array: np.ndarray) -> bytes:
        """Compress numpy array to bytes."""
        return self.compressor.compress(array.tobytes())
    
    def _decompress_array(self, data: bytes, shape: List[int], dtype: str) -> np.ndarray:
        """Decompress bytes to numpy array."""
        decompressed = self.decompressor.decompress(data)
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    
    def store_embedding(self, prompt: str, embedding: Union[torch.Tensor, Dict], 
                       model_version: str = "wan2.1-t5") -> str:
        """Store embedding with compression and deduplication."""
        with self._memory_context():
            # Handle dict format from WanVideoTextEncode
            if isinstance(embedding, dict):
                embedding_tensor = embedding.get('prompt_embeds')
                if isinstance(embedding_tensor, list):
                    embedding_tensor = embedding_tensor[0]
            else:
                embedding_tensor = embedding
            
            # Move to CPU and convert to numpy
            # Handle bfloat16 by converting to float32 first
            if embedding_tensor.dtype == torch.bfloat16:
                embedding_tensor = embedding_tensor.float()
            emb_np = embedding_tensor.detach().cpu().numpy()
            
            # Use float16 for storage efficiency
            if emb_np.dtype != np.float16:
                emb_np = emb_np.astype(np.float16)
            
            # Generate hash for deduplication
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            
            # Check if already exists
            existing = self.conn.execute(
                "SELECT id FROM embeddings WHERE prompt_hash = ?", 
                [prompt_hash]
            ).fetchone()
            
            if existing:
                # Update access stats
                self.conn.execute("""
                    UPDATE embeddings 
                    SET last_accessed = CURRENT_TIMESTAMP, 
                        access_count = access_count + 1 
                    WHERE id = ?
                """, [existing[0]])
                return existing[0]
            
            # Compress embedding
            compressed = self._compress_array(emb_np)
            compression_ratio = emb_np.nbytes / len(compressed)
            
            # Store in database
            emb_id = str(uuid.uuid4())
            self.conn.execute("""
                INSERT INTO embeddings 
                (id, prompt, prompt_hash, embedding_compressed, embedding_shape, 
                 embedding_dtype, compression_ratio, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                emb_id, prompt, prompt_hash, compressed,
                json.dumps(emb_np.shape), str(emb_np.dtype), 
                compression_ratio, model_version
            ])
            
            print(f"[EmbeddingDB] Stored embedding {emb_id[:8]} for prompt (compression: {compression_ratio:.2f}x)")
            
            # Clean up
            del embedding_tensor
            
            return emb_id
    
    def load_embedding(self, embedding_id: str, to_device: str = "cpu") -> torch.Tensor:
        """Load embedding from database."""
        result = self.conn.execute("""
            SELECT embedding_compressed, embedding_shape, embedding_dtype
            FROM embeddings 
            WHERE id = ?
        """, [embedding_id]).fetchone()
        
        if not result:
            raise ValueError(f"Embedding {embedding_id} not found")
        
        compressed, shape_json, dtype = result
        shape = json.loads(shape_json)
        
        # Decompress
        emb_np = self._decompress_array(compressed, shape, dtype)
        
        # Convert to tensor
        tensor = torch.from_numpy(emb_np).to(to_device)
        
        # Update access stats
        self.conn.execute("""
            UPDATE embeddings 
            SET last_accessed = CURRENT_TIMESTAMP, 
                access_count = access_count + 1 
            WHERE id = ?
        """, [embedding_id])
        
        return tensor
    
    def find_by_prompt(self, prompt: str) -> Optional[str]:
        """Find embedding ID by prompt."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        result = self.conn.execute(
            "SELECT id FROM embeddings WHERE prompt_hash = ?", 
            [prompt_hash]
        ).fetchone()
        
        return result[0] if result else None
    
    def vector_operation(self, operation_type: str, inputs: Dict[str, Tuple[str, float]], 
                        parameters: Dict = None) -> str:
        """
        Perform and track vector operation.
        
        Args:
            operation_type: Type of operation (add, subtract, interpolate, etc.)
            inputs: Dict of {role: (embedding_id, weight)}
            parameters: Additional operation parameters
        
        Returns:
            Result embedding ID
        """
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with self._memory_context():
            # Load input embeddings
            tensors = {}
            for role, (emb_id, weight) in inputs.items():
                tensor = self.load_embedding(emb_id, "cuda" if torch.cuda.is_available() else "cpu")
                tensors[role] = (tensor, weight)
            
            # Perform operation
            if operation_type == "add":
                result = self._op_add(tensors)
            elif operation_type == "subtract":
                result = self._op_subtract(tensors)
            elif operation_type == "interpolate":
                result = self._op_interpolate(tensors, parameters or {})
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            # Generate prompt for result
            result_prompt = f"[{operation_type}] " + " ".join(
                f"{role}:{emb_id[:8]}*{weight}" 
                for role, (emb_id, weight) in inputs.items()
            )
            
            # Store result
            result_id = self.store_embedding(result_prompt, result)
            
            # Track operation
            execution_time = int((time.time() - start_time) * 1000)
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            memory_mb = (peak_memory - start_memory) / 1024 / 1024
            
            op_id = str(uuid.uuid4())
            self.conn.execute("""
                INSERT INTO vector_operations 
                (id, operation_type, result_embedding_id, execution_time_ms, memory_peak_mb)
                VALUES (?, ?, ?, ?, ?)
            """, [op_id, operation_type, result_id, execution_time, memory_mb])
            
            # Track inputs
            for role, (emb_id, weight) in inputs.items():
                self.conn.execute("""
                    INSERT INTO operation_inputs 
                    (operation_id, embedding_id, input_role, weight)
                    VALUES (?, ?, ?, ?)
                """, [op_id, emb_id, role, weight])
            
            # Track parameters
            if parameters:
                for name, value in parameters.items():
                    self.conn.execute("""
                        INSERT INTO operation_parameters 
                        (operation_id, parameter_name, parameter_value)
                        VALUES (?, ?, ?)
                    """, [op_id, name, json.dumps(value)])
            
            print(f"[EmbeddingDB] Operation {operation_type} completed in {execution_time}ms")
            
            # Clean up tensors
            for tensor, _ in tensors.values():
                del tensor
            
            return result_id
    
    def _op_add(self, tensors: Dict[str, Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Add weighted tensors."""
        result = None
        for role, (tensor, weight) in tensors.items():
            weighted = tensor * weight
            if result is None:
                result = weighted
            else:
                result = result + weighted
        return result
    
    def _op_subtract(self, tensors: Dict[str, Tuple[torch.Tensor, float]]) -> torch.Tensor:
        """Subtract tensors (base - modifiers)."""
        base_tensor, base_weight = tensors.get('base', (None, 1.0))
        if base_tensor is None:
            raise ValueError("Subtract operation requires 'base' input")
        
        result = base_tensor * base_weight
        
        for role, (tensor, weight) in tensors.items():
            if role != 'base':
                result = result - (tensor * weight)
        
        return result
    
    def _op_interpolate(self, tensors: Dict[str, Tuple[torch.Tensor, float]], 
                       parameters: Dict) -> torch.Tensor:
        """Interpolate between tensors."""
        start_tensor, _ = tensors.get('start', (None, 1.0))
        end_tensor, _ = tensors.get('end', (None, 1.0))
        
        if start_tensor is None or end_tensor is None:
            raise ValueError("Interpolate requires 'start' and 'end' inputs")
        
        alpha = parameters.get('alpha', 0.5)
        method = parameters.get('method', 'linear')
        
        if method == 'linear':
            return (1 - alpha) * start_tensor + alpha * end_tensor
        elif method == 'spherical':
            # Spherical interpolation for normalized vectors
            start_norm = torch.nn.functional.normalize(start_tensor, dim=-1)
            end_norm = torch.nn.functional.normalize(end_tensor, dim=-1)
            
            dot = (start_norm * end_norm).sum(dim=-1, keepdim=True)
            theta = torch.acos(torch.clamp(dot, -1, 1))
            
            sin_theta = torch.sin(theta)
            factor_start = torch.sin((1 - alpha) * theta) / sin_theta
            factor_end = torch.sin(alpha * theta) / sin_theta
            
            # Handle edge case where vectors are parallel
            parallel_mask = (sin_theta.abs() < 1e-8).float()
            factor_start = factor_start * (1 - parallel_mask) + (1 - alpha) * parallel_mask
            factor_end = factor_end * (1 - parallel_mask) + alpha * parallel_mask
            
            return factor_start * start_tensor + factor_end * end_tensor
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def cleanup_old_embeddings(self, days: int = 30, min_access_count: int = 5):
        """Clean up old, rarely accessed embeddings."""
        deleted = self.conn.execute(f"""
            DELETE FROM embeddings 
            WHERE last_accessed < CURRENT_TIMESTAMP - INTERVAL '{days} days'
            AND access_count < {min_access_count}
            AND id NOT IN (
                SELECT DISTINCT embedding_id FROM operation_inputs
                UNION
                SELECT DISTINCT result_embedding_id FROM vector_operations 
                WHERE result_embedding_id IS NOT NULL
            )
        """).fetchall()
        
        print(f"[EmbeddingDB] Cleaned up {len(deleted)} old embeddings")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = {}
        
        # Total embeddings
        stats['total_embeddings'] = self.conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        
        # Total operations
        stats['total_operations'] = self.conn.execute(
            "SELECT COUNT(*) FROM vector_operations"
        ).fetchone()[0]
        
        # Storage size
        stats['storage_mb'] = os.path.getsize(self.db_path) / 1024 / 1024
        
        # Average compression ratio
        avg_compression = self.conn.execute(
            "SELECT AVG(compression_ratio) FROM embeddings"
        ).fetchone()[0]
        stats['avg_compression_ratio'] = avg_compression or 0
        
        # Operation stats
        op_stats = self.conn.execute("""
            SELECT 
                operation_type,
                COUNT(*) as count,
                AVG(execution_time_ms) as avg_time,
                AVG(memory_peak_mb) as avg_memory
            FROM vector_operations
            GROUP BY operation_type
        """).fetchall()
        
        stats['operations'] = {
            op_type: {
                'count': count,
                'avg_time_ms': avg_time,
                'avg_memory_mb': avg_memory
            }
            for op_type, count, avg_time, avg_memory in op_stats
        }
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# Global instance for easy access
_db_instance = None

def get_db() -> EmbeddingDatabase:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = EmbeddingDatabase()
    return _db_instance