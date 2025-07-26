"""
Vector arithmetic operations for WanVideo embeddings.
Implements difference extraction, arithmetic operations, and interpolation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json

from .database import get_db, EmbeddingDatabase


class WanVideoEmbeddingAnalyzer:
    """Analyze embeddings to understand their structure and properties."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "analysis_type": (["statistics", "shape", "pca_preview", "all"], {"default": "all"}),
                "store_in_db": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("analysis_report", "text_embeds")
    FUNCTION = "analyze_embedding"
    CATEGORY = "WanVideoWrapper/VectorOps"
    DESCRIPTION = "Analyze embedding structure and statistics"
    
    def analyze_embedding(self, text_embeds, analysis_type, store_in_db):
        """Analyze embedding and return report."""
        report_lines = ["=== WanVideo Embedding Analysis ===\n"]
        
        # Extract embedding tensor
        if isinstance(text_embeds, dict):
            embedding = text_embeds.get('prompt_embeds')
            if isinstance(embedding, list):
                embedding = embedding[0]
        else:
            embedding = text_embeds
        
        # Store in database if requested
        embedding_id = None
        if store_in_db:
            db = get_db()
            prompt = text_embeds.get('prompt', 'Unknown prompt') if isinstance(text_embeds, dict) else 'Analyzed embedding'
            embedding_id = db.store_embedding(prompt, text_embeds)
            report_lines.append(f"Stored in database: {embedding_id[:8]}...")
        
        # Basic shape analysis
        if analysis_type in ["shape", "all"]:
            report_lines.append("\n[Shape Information]")
            report_lines.append(f"Dimensions: {list(embedding.shape)}")
            report_lines.append(f"Total elements: {embedding.numel():,}")
            report_lines.append(f"Data type: {embedding.dtype}")
            report_lines.append(f"Device: {embedding.device}")
        
        # Statistical analysis
        if analysis_type in ["statistics", "all"]:
            report_lines.append("\n[Statistical Analysis]")
            with torch.no_grad():
                emb_cpu = embedding.cpu().float()
                report_lines.append(f"Mean: {emb_cpu.mean().item():.6f}")
                report_lines.append(f"Std Dev: {emb_cpu.std().item():.6f}")
                report_lines.append(f"Min: {emb_cpu.min().item():.6f}")
                report_lines.append(f"Max: {emb_cpu.max().item():.6f}")
                report_lines.append(f"L2 Norm: {torch.norm(emb_cpu, p=2).item():.6f}")
                
                # Sparsity
                near_zero = (emb_cpu.abs() < 1e-6).float().mean().item()
                report_lines.append(f"Near-zero values: {near_zero*100:.2f}%")
        
        # PCA preview (simplified)
        if analysis_type in ["pca_preview", "all"]:
            report_lines.append("\n[Dimensionality Preview]")
            with torch.no_grad():
                # Flatten to 2D for analysis
                emb_flat = embedding.view(-1, embedding.shape[-1]).cpu().float()
                
                # Compute variance along feature dimension
                feature_vars = emb_flat.var(dim=0)
                sorted_vars, indices = torch.sort(feature_vars, descending=True)
                
                # Show top contributing dimensions
                report_lines.append("Top 10 variance dimensions:")
                for i in range(min(10, len(sorted_vars))):
                    var_pct = (sorted_vars[i] / sorted_vars.sum()).item() * 100
                    report_lines.append(f"  Dim {indices[i].item()}: {var_pct:.2f}% variance")
                
                # Estimate intrinsic dimensionality
                cumsum = torch.cumsum(sorted_vars, dim=0)
                total_var = sorted_vars.sum()
                dims_90 = (cumsum <= 0.9 * total_var).sum().item() + 1
                dims_95 = (cumsum <= 0.95 * total_var).sum().item() + 1
                
                report_lines.append(f"\nEffective dimensionality:")
                report_lines.append(f"  90% variance: {dims_90} dimensions")
                report_lines.append(f"  95% variance: {dims_95} dimensions")
        
        # Database stats if stored
        if store_in_db and embedding_id:
            report_lines.append("\n[Database Stats]")
            db_stats = db.get_stats()
            report_lines.append(f"Total embeddings stored: {db_stats['total_embeddings']}")
            report_lines.append(f"Average compression ratio: {db_stats['avg_compression_ratio']:.2f}x")
        
        report = "\n".join(report_lines)
        return (report, text_embeds)


class WanVideoVectorDifference:
    """Calculate the difference between two embeddings."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_embeds_a": ("WANVIDEOTEXTEMBEDS",),
                "text_embeds_b": ("WANVIDEOTEXTEMBEDS",),
                "normalize": ("BOOLEAN", {"default": False}),
                "scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("difference_vector",)
    FUNCTION = "compute_difference"
    CATEGORY = "WanVideoWrapper/VectorOps"
    DESCRIPTION = "Compute A - B to extract the difference vector"
    
    def compute_difference(self, text_embeds_a, text_embeds_b, normalize, scale):
        """Compute scaled difference between embeddings."""
        print("\n[VectorDifference] Computing difference vector")
        
        # Extract tensors
        tensor_a = self._extract_tensor(text_embeds_a)
        tensor_b = self._extract_tensor(text_embeds_b)
        
        # Ensure compatible shapes
        if tensor_a.shape != tensor_b.shape:
            print(f"[VectorDifference] Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}")
            
            # Check for dimension size mismatch
            if (tensor_a.shape[-1] == 4096 and tensor_b.shape[-1] == 5120) or \
               (tensor_a.shape[-1] == 5120 and tensor_b.shape[-1] == 4096):
                print("[VectorDifference] WARNING: Mixing raw T5 embeddings (4096) with processed embeddings (5120)")
                print("[VectorDifference] This may produce unexpected results. Consider using matching embedding types.")
            
            # Pad or truncate to match
            tensor_a, tensor_b = self._align_tensors(tensor_a, tensor_b)
            print(f"[VectorDifference] Aligned shapes: {tensor_a.shape} and {tensor_b.shape}")
        
        # Ensure tensors are on the same device
        if tensor_a.device != tensor_b.device:
            print(f"[VectorDifference] Device mismatch: {tensor_a.device} vs {tensor_b.device}")
            # Move tensor_b to tensor_a's device
            tensor_b = tensor_b.to(device=tensor_a.device, dtype=tensor_a.dtype)
            print(f"[VectorDifference] Moved tensor_b to {tensor_a.device}")
        
        # Compute difference
        with torch.no_grad():
            difference = tensor_a - tensor_b
            
            if normalize:
                # Normalize the difference vector
                norm = torch.norm(difference, p=2, dim=-1, keepdim=True)
                difference = difference / (norm + 1e-8)
                print("[VectorDifference] Normalized difference vector")
            
            if scale != 1.0:
                difference = difference * scale
                print(f"[VectorDifference] Scaled by {scale}")
        
        # Create output embedding dict
        result_embeds = {
            'prompt_embeds': [difference],
            'prompt': f"[Difference] A-B scaled by {scale}",
            'operation': 'difference',
            'source_a': text_embeds_a.get('prompt', 'Unknown') if isinstance(text_embeds_a, dict) else 'Embedding A',
            'source_b': text_embeds_b.get('prompt', 'Unknown') if isinstance(text_embeds_b, dict) else 'Embedding B'
        }
        
        # Store in database
        db = get_db()
        db.store_embedding(result_embeds['prompt'], result_embeds)
        
        print("[VectorDifference] Difference computed successfully")
        return (result_embeds,)
    
    def _extract_tensor(self, text_embeds):
        """Extract tensor from embedding dict or tensor."""
        if isinstance(text_embeds, dict):
            # Check for latent embeddings first
            if 'latent' in text_embeds:
                tensor = text_embeds['latent']
                # Move to GPU if needed (latent embeddings are stored on CPU)
                if tensor.device.type == 'cpu' and torch.cuda.is_available():
                    # Get device/dtype info if available
                    device_str = text_embeds.get('device', 'cuda:0')
                    dtype_str = text_embeds.get('dtype', 'torch.float16')
                    
                    # Parse device
                    device = torch.device(device_str)
                    
                    # Parse dtype
                    if 'float16' in dtype_str:
                        dtype = torch.float16
                    elif 'float32' in dtype_str:
                        dtype = torch.float32
                    elif 'e4m3fn' in dtype_str:
                        dtype = torch.float8_e4m3fn
                    elif 'e5m2' in dtype_str:
                        dtype = torch.float8_e5m2
                    else:
                        dtype = torch.float16  # default
                    
                    tensor = tensor.to(device=device, dtype=dtype)
                return tensor
            # Otherwise check for prompt_embeds
            embedding = text_embeds.get('prompt_embeds')
            if isinstance(embedding, list):
                return embedding[0]
            return embedding
        return text_embeds
    
    def _align_tensors(self, tensor_a, tensor_b):
        """Align tensor shapes by padding or truncating."""
        # Handle dimension mismatch
        if tensor_a.ndim != tensor_b.ndim:
            # Convert 2D to 3D if needed by adding batch dimension
            if tensor_a.ndim == 2 and tensor_b.ndim == 3:
                tensor_a = tensor_a.unsqueeze(0)  # Add batch dimension
            elif tensor_a.ndim == 3 and tensor_b.ndim == 2:
                tensor_b = tensor_b.unsqueeze(0)  # Add batch dimension
        
        # Get target shape (use larger dimensions)
        target_shape = []
        for dim_a, dim_b in zip(tensor_a.shape, tensor_b.shape):
            target_shape.append(max(dim_a, dim_b))
        
        # Pad or truncate each tensor
        aligned_a = self._resize_tensor(tensor_a, target_shape)
        aligned_b = self._resize_tensor(tensor_b, target_shape)
        
        return aligned_a, aligned_b
    
    def _resize_tensor(self, tensor, target_shape):
        """Resize tensor to target shape."""
        if list(tensor.shape) == target_shape:
            return tensor
        
        # Handle 2D tensors [seq_len, dim]
        if len(tensor.shape) == 2 and len(target_shape) == 2:
            seq_len, dim = tensor.shape
            target_seq, target_dim = target_shape
            
            # Handle sequence length
            if seq_len < target_seq:
                # Pad with zeros
                padding = torch.zeros(target_seq - seq_len, dim, 
                                    dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=0)
            elif seq_len > target_seq:
                # Truncate
                tensor = tensor[:target_seq, :]
            
            # Handle dimension mismatch
            if dim != target_dim:
                if dim < target_dim:
                    # Pad embedding dimension
                    padding = torch.zeros(tensor.shape[0], target_dim - dim,
                                        dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=1)
                else:
                    # Truncate embedding dimension
                    tensor = tensor[:, :target_dim]
                
        # Handle 3D tensors [batch, seq_len, dim]
        elif len(tensor.shape) == 3 and len(target_shape) == 3:
            batch, seq_len, dim = tensor.shape
            target_batch, target_seq, target_dim = target_shape
            
            # Handle sequence length
            if seq_len < target_seq:
                # Pad with zeros
                padding = torch.zeros(batch, target_seq - seq_len, dim, 
                                    dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=1)
            elif seq_len > target_seq:
                # Truncate
                tensor = tensor[:, :target_seq, :]
            
            # Handle dimension mismatch
            if dim != target_dim:
                if dim < target_dim:
                    # Pad embedding dimension
                    padding = torch.zeros(batch, tensor.shape[1], target_dim - dim,
                                        dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=2)
                else:
                    # Truncate embedding dimension
                    tensor = tensor[:, :, :target_dim]
        
        return tensor


class WanVideoVectorArithmetic:
    """Perform arithmetic operations on embeddings."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_embeds": ("WANVIDEOTEXTEMBEDS",),
                "operation": (["add", "weighted_sum", "multiply", "normalize"], {"default": "add"}),
            },
            "optional": {
                "operand_1": ("WANVIDEOTEXTEMBEDS",),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "operand_2": ("WANVIDEOTEXTEMBEDS",),
                "weight_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "operand_3": ("WANVIDEOTEXTEMBEDS",),
                "weight_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("result_embeds",)
    FUNCTION = "perform_arithmetic"
    CATEGORY = "WanVideoWrapper/VectorOps"
    DESCRIPTION = "Perform arithmetic operations on multiple embeddings"
    
    def perform_arithmetic(self, base_embeds, operation, **kwargs):
        """Perform arithmetic operation on embeddings."""
        print(f"\n[VectorArithmetic] Performing {operation} operation")
        
        # Collect operands
        operands = [(base_embeds, 1.0)]  # Base always has weight 1.0
        
        for i in range(1, 4):
            operand_key = f"operand_{i}"
            weight_key = f"weight_{i}"
            if operand_key in kwargs and kwargs[operand_key] is not None:
                operands.append((kwargs[operand_key], kwargs.get(weight_key, 1.0)))
        
        print(f"[VectorArithmetic] Processing {len(operands)} operands")
        
        # Extract and align tensors
        tensors = []
        max_shape = None
        
        for embeds, weight in operands:
            tensor = self._extract_tensor(embeds)
            if max_shape is None:
                max_shape = list(tensor.shape)
            else:
                for i, (curr, new) in enumerate(zip(max_shape, tensor.shape)):
                    max_shape[i] = max(curr, new)
            tensors.append((tensor, weight))
        
        # Align all tensors to max shape
        aligned_tensors = []
        for tensor, weight in tensors:
            aligned = self._resize_tensor(tensor, max_shape)
            aligned_tensors.append((aligned, weight))
        
        # Perform operation
        with torch.no_grad():
            if operation == "add":
                result = sum(tensor * weight for tensor, weight in aligned_tensors)
            
            elif operation == "weighted_sum":
                # Normalize weights
                total_weight = sum(abs(weight) for _, weight in aligned_tensors)
                if total_weight > 0:
                    result = sum(tensor * (weight / total_weight) 
                               for tensor, weight in aligned_tensors)
                else:
                    result = aligned_tensors[0][0]
            
            elif operation == "multiply":
                result = aligned_tensors[0][0] * aligned_tensors[0][1]
                for tensor, weight in aligned_tensors[1:]:
                    result = result * (tensor * weight)
            
            elif operation == "normalize":
                # Sum and normalize
                result = sum(tensor * weight for tensor, weight in aligned_tensors)
                norm = torch.norm(result, p=2, dim=-1, keepdim=True)
                result = result / (norm + 1e-8)
            
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Create result embedding
        result_embeds = {
            'prompt_embeds': [result],
            'prompt': f"[{operation}] {len(operands)} operands",
            'operation': operation,
            'operand_count': len(operands)
        }
        
        # Store in database with tracking
        db = get_db()
        
        # Store result and track operation
        if len(operands) > 1:
            # First store intermediate operands if needed
            operand_ids = []
            for i, (embeds, weight) in enumerate(operands):
                prompt = embeds.get('prompt', f'Operand {i}') if isinstance(embeds, dict) else f'Operand {i}'
                emb_id = db.store_embedding(prompt, embeds)
                operand_ids.append((emb_id, weight))
            
            # Perform tracked operation
            inputs = {f"operand_{i}": (emb_id, weight) 
                     for i, (emb_id, weight) in enumerate(operand_ids)}
            result_id = db.vector_operation(operation, inputs, {"method": operation})
            result_embeds['embedding_id'] = result_id
        else:
            # Just store the result
            db.store_embedding(result_embeds['prompt'], result_embeds)
        
        print(f"[VectorArithmetic] {operation} completed")
        return (result_embeds,)
    
    def _extract_tensor(self, text_embeds):
        """Extract tensor from embedding dict or tensor."""
        if isinstance(text_embeds, dict):
            # Check for latent embeddings first
            if 'latent' in text_embeds:
                tensor = text_embeds['latent']
                # Move to GPU if needed (latent embeddings are stored on CPU)
                if tensor.device.type == 'cpu' and torch.cuda.is_available():
                    # Get device/dtype info if available
                    device_str = text_embeds.get('device', 'cuda:0')
                    dtype_str = text_embeds.get('dtype', 'torch.float16')
                    
                    # Parse device
                    device = torch.device(device_str)
                    
                    # Parse dtype
                    if 'float16' in dtype_str:
                        dtype = torch.float16
                    elif 'float32' in dtype_str:
                        dtype = torch.float32
                    elif 'e4m3fn' in dtype_str:
                        dtype = torch.float8_e4m3fn
                    elif 'e5m2' in dtype_str:
                        dtype = torch.float8_e5m2
                    else:
                        dtype = torch.float16  # default
                    
                    tensor = tensor.to(device=device, dtype=dtype)
                return tensor
            # Otherwise check for prompt_embeds
            embedding = text_embeds.get('prompt_embeds')
            if isinstance(embedding, list):
                return embedding[0]
            return embedding
        return text_embeds
    
    def _resize_tensor(self, tensor, target_shape):
        """Resize tensor to target shape."""
        if list(tensor.shape) == target_shape:
            return tensor
        
        # For sequence length dimension, pad or truncate
        if len(tensor.shape) == 3:  # [batch, seq_len, dim]
            batch, seq_len, dim = tensor.shape
            target_batch, target_seq, target_dim = target_shape
            
            # Handle sequence length
            if seq_len < target_seq:
                # Pad with zeros
                padding = torch.zeros(batch, target_seq - seq_len, dim, 
                                    dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=1)
            elif seq_len > target_seq:
                # Truncate
                tensor = tensor[:, :target_seq, :]
        
        return tensor


class WanVideoVectorInterpolation:
    """Interpolate between embeddings with various methods."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_embeds": ("WANVIDEOTEXTEMBEDS",),
                "end_embeds": ("WANVIDEOTEXTEMBEDS",),
                "steps": ("INT", {"default": 5, "min": 2, "max": 20, "step": 1}),
                "method": (["linear", "spherical", "cubic"], {"default": "linear"}),
                "include_endpoints": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("interpolated_sequence",)
    FUNCTION = "interpolate"
    CATEGORY = "WanVideoWrapper/VectorOps"
    DESCRIPTION = "Create smooth interpolation between embeddings"
    
    def interpolate(self, start_embeds, end_embeds, steps, method, include_endpoints):
        """Interpolate between embeddings."""
        print(f"\n[VectorInterpolation] Creating {steps}-step {method} interpolation")
        
        # Extract tensors
        start_tensor = self._extract_tensor(start_embeds)
        end_tensor = self._extract_tensor(end_embeds)
        
        # Align shapes
        if start_tensor.shape != end_tensor.shape:
            start_tensor, end_tensor = self._align_tensors(start_tensor, end_tensor)
        
        # Generate interpolation steps
        if include_endpoints:
            alphas = torch.linspace(0, 1, steps)
        else:
            alphas = torch.linspace(0, 1, steps + 2)[1:-1]
        
        interpolated_tensors = []
        
        with torch.no_grad():
            for i, alpha in enumerate(alphas):
                alpha_val = alpha.item()
                
                if method == "linear":
                    interp = (1 - alpha_val) * start_tensor + alpha_val * end_tensor
                
                elif method == "spherical":
                    # Spherical interpolation
                    start_norm = torch.nn.functional.normalize(start_tensor, dim=-1)
                    end_norm = torch.nn.functional.normalize(end_tensor, dim=-1)
                    
                    dot = (start_norm * end_norm).sum(dim=-1, keepdim=True)
                    theta = torch.acos(torch.clamp(dot, -1, 1))
                    
                    sin_theta = torch.sin(theta)
                    factor_start = torch.sin((1 - alpha_val) * theta) / (sin_theta + 1e-8)
                    factor_end = torch.sin(alpha_val * theta) / (sin_theta + 1e-8)
                    
                    # Handle parallel vectors
                    parallel_mask = (sin_theta.abs() < 1e-8)
                    factor_start = torch.where(parallel_mask, 1 - alpha_val, factor_start)
                    factor_end = torch.where(parallel_mask, alpha_val, factor_end)
                    
                    interp = factor_start * start_tensor + factor_end * end_tensor
                
                elif method == "cubic":
                    # Cubic interpolation (smoothstep)
                    t = alpha_val
                    smooth_t = t * t * (3 - 2 * t)
                    interp = (1 - smooth_t) * start_tensor + smooth_t * end_tensor
                
                interpolated_tensors.append(interp)
                
                # Store intermediate step
                db = get_db()
                step_embeds = {
                    'prompt_embeds': [interp],
                    'prompt': f"[Interpolation {method}] Step {i+1}/{steps} (α={alpha_val:.2f})"
                }
                db.store_embedding(step_embeds['prompt'], step_embeds)
        
        # Create sequence embedding (using the middle point as representative)
        mid_idx = len(interpolated_tensors) // 2
        result_embeds = {
            'prompt_embeds': [interpolated_tensors[mid_idx]],
            'prompt': f"[Interpolation {method}] {steps} steps",
            'operation': 'interpolation',
            'method': method,
            'steps': steps,
            'interpolated_sequence': interpolated_tensors  # Store full sequence
        }
        
        print(f"[VectorInterpolation] Generated {len(interpolated_tensors)} interpolation steps")
        return (result_embeds,)
    
    def _extract_tensor(self, text_embeds):
        """Extract tensor from embedding dict or tensor."""
        if isinstance(text_embeds, dict):
            # Check for latent embeddings first
            if 'latent' in text_embeds:
                tensor = text_embeds['latent']
                # Move to GPU if needed (latent embeddings are stored on CPU)
                if tensor.device.type == 'cpu' and torch.cuda.is_available():
                    # Get device/dtype info if available
                    device_str = text_embeds.get('device', 'cuda:0')
                    dtype_str = text_embeds.get('dtype', 'torch.float16')
                    
                    # Parse device
                    device = torch.device(device_str)
                    
                    # Parse dtype
                    if 'float16' in dtype_str:
                        dtype = torch.float16
                    elif 'float32' in dtype_str:
                        dtype = torch.float32
                    elif 'e4m3fn' in dtype_str:
                        dtype = torch.float8_e4m3fn
                    elif 'e5m2' in dtype_str:
                        dtype = torch.float8_e5m2
                    else:
                        dtype = torch.float16  # default
                    
                    tensor = tensor.to(device=device, dtype=dtype)
                return tensor
            # Otherwise check for prompt_embeds
            embedding = text_embeds.get('prompt_embeds')
            if isinstance(embedding, list):
                return embedding[0]
            return embedding
        return text_embeds
    
    def _align_tensors(self, tensor_a, tensor_b):
        """Align tensor shapes by padding or truncating."""
        # Get target shape (use larger dimensions)
        target_shape = []
        for dim_a, dim_b in zip(tensor_a.shape, tensor_b.shape):
            target_shape.append(max(dim_a, dim_b))
        
        # Resize both tensors
        def resize(tensor, shape):
            if list(tensor.shape) == shape:
                return tensor
            
            if len(tensor.shape) == 3:  # [batch, seq_len, dim]
                batch, seq_len, dim = tensor.shape
                target_batch, target_seq, target_dim = shape
                
                if seq_len < target_seq:
                    padding = torch.zeros(batch, target_seq - seq_len, dim, 
                                        dtype=tensor.dtype, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=1)
                elif seq_len > target_seq:
                    tensor = tensor[:, :target_seq, :]
            
            return tensor
        
        return resize(tensor_a, target_shape), resize(tensor_b, target_shape)


class WanVideoEmbeddingDatabase:
    """Database management node for embeddings."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["stats", "cleanup", "list_recent"], {"default": "stats"}),
                "cleanup_days": ("INT", {"default": 30, "min": 1, "max": 365}),
                "cleanup_min_access": ("INT", {"default": 5, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "manage_database"
    CATEGORY = "WanVideoWrapper/VectorOps"
    DESCRIPTION = "Manage embedding database"
    
    def manage_database(self, action, cleanup_days, cleanup_min_access):
        """Perform database management action."""
        db = get_db()
        report_lines = [f"=== Embedding Database {action.title()} ===\n"]
        
        if action == "stats":
            stats = db.get_stats()
            report_lines.append(f"Total embeddings: {stats['total_embeddings']}")
            report_lines.append(f"Total operations: {stats['total_operations']}")
            report_lines.append(f"Storage size: {stats['storage_mb']:.2f} MB")
            report_lines.append(f"Average compression: {stats['avg_compression_ratio']:.2f}x")
            
            if stats['operations']:
                report_lines.append("\nOperation Statistics:")
                for op_type, op_stats in stats['operations'].items():
                    report_lines.append(f"  {op_type}:")
                    report_lines.append(f"    Count: {op_stats['count']}")
                    report_lines.append(f"    Avg time: {op_stats['avg_time_ms']:.1f}ms")
                    report_lines.append(f"    Avg memory: {op_stats['avg_memory_mb']:.1f}MB")
        
        elif action == "cleanup":
            db.cleanup_old_embeddings(cleanup_days, cleanup_min_access)
            report_lines.append(f"Cleaned embeddings older than {cleanup_days} days")
            report_lines.append(f"with less than {cleanup_min_access} accesses")
            
            # Show new stats
            stats = db.get_stats()
            report_lines.append(f"\nRemaining embeddings: {stats['total_embeddings']}")
            report_lines.append(f"Storage size: {stats['storage_mb']:.2f} MB")
        
        elif action == "list_recent":
            # Query recent embeddings
            recent = db.conn.execute("""
                SELECT prompt, created_at, access_count, compression_ratio
                FROM embeddings
                ORDER BY created_at DESC
                LIMIT 10
            """).fetchall()
            
            report_lines.append("Recent Embeddings:")
            for prompt, created, accesses, compression in recent:
                # Truncate long prompts
                display_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
                report_lines.append(f"\n  {display_prompt}")
                report_lines.append(f"    Created: {created}")
                report_lines.append(f"    Accesses: {accesses}, Compression: {compression:.2f}x")
        
        return ("\n".join(report_lines),)


# Update node mappings
VECTOR_NODE_CLASS_MAPPINGS = {
    "WanVideoEmbeddingAnalyzer": WanVideoEmbeddingAnalyzer,
    "WanVideoVectorDifference": WanVideoVectorDifference,
    "WanVideoVectorArithmetic": WanVideoVectorArithmetic,
    "WanVideoVectorInterpolation": WanVideoVectorInterpolation,
    "WanVideoEmbeddingDatabase": WanVideoEmbeddingDatabase,
}

VECTOR_NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEmbeddingAnalyzer": "WanVideo Embedding Analyzer",
    "WanVideoVectorDifference": "WanVideo Vector Difference",
    "WanVideoVectorArithmetic": "WanVideo Vector Arithmetic",
    "WanVideoVectorInterpolation": "WanVideo Vector Interpolation",
    "WanVideoEmbeddingDatabase": "WanVideo Embedding Database",
}