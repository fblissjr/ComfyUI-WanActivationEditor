"""
Runtime debugging utilities for WanVideo Activation Editor
Can be imported and used anywhere for debugging
"""

import torch
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List
import traceback

# Global debug flags
DEBUG = os.environ.get('WAN_ACTIVATION_DEBUG', '0') == '1'
VERBOSE = os.environ.get('WAN_ACTIVATION_VERBOSE', '0') == '1'
TRACE = os.environ.get('WAN_ACTIVATION_TRACE', '0') == '1'

# DEFAULT: Verbose logging is OFF
# To enable verbose logging, either:
# 1. Set log_level to "verbose" in WanVideoActivationEditor node
# 2. Set environment variable: export WAN_ACTIVATION_VERBOSE=1
# 3. Uncomment ONE line below:
# DEBUG = True      # Basic debug output only
# VERBOSE = True    # Verbose output (automatically enables DEBUG too)

# Runtime log level override
_LOG_LEVEL_OVERRIDE = None

def set_log_level(level: str):
    """Set the runtime log level. Levels: 'off', 'basic', 'verbose', 'trace'"""
    global _LOG_LEVEL_OVERRIDE, DEBUG, VERBOSE, TRACE
    _LOG_LEVEL_OVERRIDE = level
    
    if level == "off":
        DEBUG = False
        VERBOSE = False
        TRACE = False
    elif level == "basic":
        DEBUG = True
        VERBOSE = False
        TRACE = False
    elif level == "verbose":
        DEBUG = True
        VERBOSE = True
        TRACE = False
    elif level == "trace":
        DEBUG = True
        VERBOSE = True
        TRACE = True
    
    # Immediate feedback
    if level != "off":
        print(f"[WanActivationEditor] Log level set to: {level}")

def get_log_level() -> str:
    """Get current log level"""
    if _LOG_LEVEL_OVERRIDE:
        return _LOG_LEVEL_OVERRIDE
    elif TRACE:
        return "trace"
    elif VERBOSE:
        return "verbose"
    elif DEBUG:
        return "basic"
    else:
        return "off"

def debug_print(message: str, level: str = "INFO"):
    """Print debug message if debugging is enabled"""
    if DEBUG or VERBOSE:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[WAN_DEBUG {timestamp}] [{level}] {message}")

def verbose_print(message: str):
    """Print verbose message if verbose mode is enabled"""
    if VERBOSE:
        debug_print(message, "VERBOSE")

def trace_print(message: str):
    """Print trace message if trace mode is enabled"""
    if TRACE:
        debug_print(message, "TRACE")
        # Also print stack trace in trace mode
        stack = traceback.extract_stack()[:-1]
        for frame in stack[-3:]:  # Last 3 frames
            print(f"  -> {frame.filename}:{frame.lineno} in {frame.name}")

def debug_tensor(tensor: torch.Tensor, name: str = "tensor", detailed: bool = False):
    """Debug print tensor information"""
    if not (DEBUG or VERBOSE):
        return

    info = [
        f"{name}:",
        f"  shape: {tensor.shape}",
        f"  device: {tensor.device}",
        f"  dtype: {tensor.dtype}",
    ]

    if detailed or VERBOSE:
        info.extend([
            f"  mean: {tensor.mean().item():.6f}",
            f"  std: {tensor.std().item():.6f}",
            f"  min: {tensor.min().item():.6f}",
            f"  max: {tensor.max().item():.6f}",
            f"  norm: {torch.norm(tensor).item():.6f}",
        ])

        # Check for special values
        if torch.isnan(tensor).any():
            info.append(f"  WARNING: Contains NaN values!")
        if torch.isinf(tensor).any():
            info.append(f"  WARNING: Contains Inf values!")
        if (tensor == 0).all():
            info.append(f"  WARNING: All zeros!")

    debug_print("\n".join(info))

def debug_config(config: Dict[str, Any], name: str = "config"):
    """Debug print configuration dictionary"""
    if not (DEBUG or VERBOSE):
        return

    debug_print(f"{name}: {json.dumps(config, indent=2, default=str)}")

def debug_injection_point(block_idx: int, context: torch.Tensor,
                         injection: Optional[torch.Tensor] = None,
                         strength: float = 0.0):
    """Debug print at injection point"""
    if not (DEBUG or VERBOSE or TRACE):
        return

    debug_print(f"\n=== Injection Point: Block {block_idx} ===")
    debug_tensor(context, f"context (block {block_idx})")

    if injection is not None:
        debug_tensor(injection, f"injection (block {block_idx})")

        # Calculate difference
        if context.shape == injection.shape:
            diff = (context - injection).abs()
            diff_percent = (diff > 0.01).float().mean().item() * 100
            debug_print(f"  difference: {diff.mean().item():.6f} ({diff_percent:.1f}% different)")
        else:
            debug_print(f"  WARNING: Shape mismatch! {context.shape} vs {injection.shape}")

    debug_print(f"  strength: {strength}")

def debug_memory(prefix: str = ""):
    """Debug print memory usage"""
    if not (DEBUG or VERBOSE):
        return

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        debug_print(f"{prefix}CUDA Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

def create_debug_checkpoint(name: str, data: Dict[str, Any]):
    """Create a debug checkpoint that can be loaded later"""
    if not (DEBUG or VERBOSE):
        return

    checkpoint = {
        'name': name,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }

    # Save to temp file if TRACE mode
    if TRACE:
        filename = f"/tmp/wan_debug_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
            trace_print(f"Debug checkpoint saved to: {filename}")
        except Exception as e:
            debug_print(f"Failed to save checkpoint: {e}", "ERROR")

def assert_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
    """Assert tensor has expected shape, with helpful debug info"""
    if tensor.shape != expected_shape:
        error_msg = f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
        debug_print(error_msg, "ERROR")
        debug_tensor(tensor, name, detailed=True)
        raise ValueError(error_msg)

def compare_embeddings(embed1: torch.Tensor, embed2: torch.Tensor,
                      name1: str = "embed1", name2: str = "embed2") -> Dict[str, float]:
    """Compare two embeddings and return statistics"""
    stats = {}

    if embed1.shape != embed2.shape:
        debug_print(f"Shape mismatch: {name1}={embed1.shape} vs {name2}={embed2.shape}", "WARNING")
        return stats

    # Move to same device for comparison
    if embed1.device != embed2.device:
        embed2 = embed2.to(embed1.device)

    # Calculate statistics
    diff = (embed1 - embed2).abs()
    stats['mean_diff'] = diff.mean().item()
    stats['max_diff'] = diff.max().item()
    stats['percent_different'] = (diff > 0.01).float().mean().item() * 100

    # Cosine similarity
    embed1_flat = embed1.flatten()
    embed2_flat = embed2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(embed1_flat.unsqueeze(0),
                                                    embed2_flat.unsqueeze(0))
    stats['cosine_similarity'] = cos_sim.item()

    if DEBUG or VERBOSE:
        debug_print(f"\nComparison {name1} vs {name2}:")
        for key, value in stats.items():
            debug_print(f"  {key}: {value:.4f}")

    return stats

# Decorators for function debugging
def debug_function(func):
    """Decorator to debug function calls"""
    def wrapper(*args, **kwargs):
        if DEBUG or VERBOSE:
            debug_print(f"Calling {func.__name__}")
            if VERBOSE:
                debug_print(f"  args: {args}")
                debug_print(f"  kwargs: {kwargs}")

        result = func(*args, **kwargs)

        if DEBUG or VERBOSE:
            debug_print(f"Completed {func.__name__}")

        return result
    return wrapper

def trace_function(func):
    """Decorator for detailed function tracing"""
    def wrapper(*args, **kwargs):
        if TRACE:
            trace_print(f"ENTER {func.__name__}")
            start_time = datetime.now()

        try:
            result = func(*args, **kwargs)

            if TRACE:
                elapsed = (datetime.now() - start_time).total_seconds()
                trace_print(f"EXIT {func.__name__} (took {elapsed:.3f}s)")

            return result
        except Exception as e:
            if TRACE:
                trace_print(f"ERROR in {func.__name__}: {e}")
            raise

    return wrapper

# Context manager for debugging blocks
class DebugBlock:
    def __init__(self, name: str):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if DEBUG or VERBOSE:
            debug_print(f"=== BEGIN {self.name} ===")
            self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DEBUG or VERBOSE:
            elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            if exc_type is None:
                debug_print(f"=== END {self.name} (took {elapsed:.3f}s) ===")
            else:
                debug_print(f"=== ERROR in {self.name}: {exc_val} ===", "ERROR")

        if exc_type is not None and TRACE:
            traceback.print_exc()
