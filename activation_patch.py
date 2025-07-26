"""
Runtime patching for WanVideo transformer blocks.
Enables block-level activation editing without modifying WanVideoWrapper.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import weakref
import os
import math
import sys

# Import debug utilities
from .debug_utils import (
    debug_print, verbose_print, trace_print, debug_tensor,
    debug_injection_point, debug_memory, DebugBlock,
    debug_function, trace_function, DEBUG, VERBOSE
)


class ActivationPatcher:
    """Manages runtime patching of transformer blocks for activation editing."""
    
    def __init__(self):
        self.active_patches = weakref.WeakKeyDictionary()
        self._text_embedding_layer = None
    
    def patch_model(self, model_patcher, activation_config: Dict[str, Any]):
        """
        Patch transformer blocks to enable activation editing.
        
        Args:
            model_patcher: ComfyUI ModelPatcher instance
            activation_config: Configuration dict with:
                - active_blocks: List of block indices to modify
                - injection_strength: Blend strength (0-1)
                - injection_embeds: Alternative embeddings to inject
        """
        try:
            # Access the actual model inside ComfyUI's ModelPatcher
            actual_model = self._find_model(model_patcher)
            if actual_model is None:
                raise RuntimeError("Could not access model from ModelPatcher")
            
            # Find transformer blocks
            transformer_blocks = self._find_transformer_blocks(actual_model)
            if transformer_blocks is None:
                debug_print("ERROR: Failed to locate transformer blocks!")
                raise RuntimeError("Failed to locate transformer blocks for patching")
            
            debug_print(f"Found {len(transformer_blocks)} transformer blocks")
            debug_print(f"Transformer blocks type: {type(transformer_blocks)}")
            debug_print(f"First block type: {type(transformer_blocks[0]) if len(transformer_blocks) > 0 else 'empty'}")
            
            # Find text_embedding layer for dimension projection
            self._find_text_embedding_layer(actual_model)
            
            # Store original forward methods
            if model_patcher not in self.active_patches:
                self.active_patches[model_patcher] = {
                    'original_forwards': {},
                    'config': activation_config
                }
                debug_print("Created new patch entry for model")
            else:
                debug_print("Model already has patch entry")
            
            active_blocks = activation_config.get('active_blocks', [])
            injection_strength = activation_config.get('injection_strength', 0.5)
            injection_embeds = activation_config.get('injection_embeds')
            
            debug_print(f"Config - active_blocks: {active_blocks}")
            debug_print(f"Config - injection_strength: {injection_strength}")
            debug_print(f"Config - injection_embeds type: {type(injection_embeds)}")
            if injection_embeds is not None:
                if hasattr(injection_embeds, 'shape'):
                    debug_print(f"Config - injection_embeds shape: {injection_embeds.shape}")
                elif isinstance(injection_embeds, dict):
                    debug_print(f"Config - injection_embeds keys: {list(injection_embeds.keys())}")
            
            # Patch each active block
            patched_count = 0
            debug_print(f"Starting to patch {len(active_blocks)} blocks...")
            for block_idx in active_blocks:
                if block_idx >= len(transformer_blocks):
                    debug_print(f"Warning: Block {block_idx} exceeds available blocks ({len(transformer_blocks)})")
                    continue
                
                debug_print(f"Patching block {block_idx}...")
                
                block = transformer_blocks[block_idx]
                
                # Store original if not already stored
                if block_idx not in self.active_patches[model_patcher]['original_forwards']:
                    self.active_patches[model_patcher]['original_forwards'][block_idx] = block.forward
                
                # Create patched forward method
                original_forward = self.active_patches[model_patcher]['original_forwards'][block_idx]
                patched_forward = self._create_patched_forward(block_idx, original_forward, injection_strength, injection_embeds)
                
                # Store reference to help with debugging
                patched_forward._block_idx = block_idx
                patched_forward._is_wan_patch = True
                
                block.forward = patched_forward
                
                # Verify patch was applied
                if block.forward == patched_forward:
                    debug_print(f"✓ Block {block_idx} patched successfully")
                    debug_print(f"  Forward method type: {type(block.forward)}")
                    debug_print(f"  Has injection embeds: {injection_embeds is not None}")
                    if injection_embeds is not None:
                        if hasattr(injection_embeds, 'shape'):
                            debug_print(f"  Injection shape: {injection_embeds.shape}")
                        elif isinstance(injection_embeds, dict):
                            debug_print(f"  Injection is dict with keys: {list(injection_embeds.keys())}")
                    patched_count += 1
                else:
                    debug_print(f"✗ Block {block_idx} patch failed!")
            
            debug_print(f"Successfully patched {patched_count}/{len(active_blocks)} blocks")
            
            # Add a test to ensure patches will work
            if patched_count > 0 and injection_embeds is not None:
                debug_print(f"Injection embeddings ready: shape={injection_embeds.shape if hasattr(injection_embeds, 'shape') else 'unknown'}")
            
            # Return success status
            return patched_count > 0
        
        except Exception as e:
            debug_print(f"ERROR in patch_model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unpatch_model(self, model_patcher):
        """Remove patches from a model."""
        if model_patcher not in self.active_patches:
            return
        
        try:
            # Access the model
            actual_model = self._find_model(model_patcher)
            if actual_model is None:
                return
            
            # Find transformer blocks
            transformer_blocks = self._find_transformer_blocks(actual_model)
            if transformer_blocks is None:
                return
            
            # Restore original forwards
            patch_info = self.active_patches[model_patcher]
            for block_idx, original_forward in patch_info['original_forwards'].items():
                if block_idx < len(transformer_blocks):
                    transformer_blocks[block_idx].forward = original_forward
                    verbose_print(f"Restored block {block_idx}")
            
            # Clean up
            del self.active_patches[model_patcher]
            debug_print("Model unpatched")
            
        except Exception as e:
            debug_print(f"Error during unpatching: {e}")
    
    def _find_model(self, model_patcher):
        """Find the actual model inside the ModelPatcher."""
        # Try multiple ways to access the model
        # WanVideoWrapper uses model.model.diffusion_model
        if hasattr(model_patcher, 'model'):
            model = model_patcher.model
            if hasattr(model, 'diffusion_model'):
                debug_print("Found model at model_patcher.model.diffusion_model")
                return model.diffusion_model
            elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                debug_print("Found model at model_patcher.model.model.diffusion_model")
                return model.model.diffusion_model
            else:
                debug_print("Found model at model_patcher.model")
                return model
        elif hasattr(model_patcher, 'transformer_blocks') or hasattr(model_patcher, 'blocks'):
            debug_print("Model patcher appears to be the model itself")
            return model_patcher
        return None
    
    def _find_transformer_blocks(self, model):
        """Find transformer blocks in the model."""
        # Check various possible paths to transformer blocks
        # WanVideoWrapper uses model.blocks
        paths_to_try = [
            lambda m: m.blocks if hasattr(m, 'blocks') else None,
            lambda m: m.transformer_blocks if hasattr(m, 'transformer_blocks') else None,
            lambda m: m.transformer.transformer_blocks if hasattr(m, 'transformer') and hasattr(m.transformer, 'transformer_blocks') else None,
            lambda m: m.transformer.blocks if hasattr(m, 'transformer') and hasattr(m.transformer, 'blocks') else None,
            lambda m: m.model.transformer_blocks if hasattr(m, 'model') and hasattr(m.model, 'transformer_blocks') else None,
            lambda m: m.model.blocks if hasattr(m, 'model') and hasattr(m.model, 'blocks') else None,
            lambda m: m._modules.get('transformer_blocks') if hasattr(m, '_modules') else None,
            lambda m: m._modules.get('blocks') if hasattr(m, '_modules') else None,
        ]
        
        for path_func in paths_to_try:
            try:
                result = path_func(model)
                if result is not None and hasattr(result, '__len__'):
                    debug_print(f"Found blocks at: {path_func.__code__.co_consts[0]}")
                    return result
            except (AttributeError, TypeError):
                continue
        
        return None
    
    def _find_text_embedding_layer(self, model):
        """Find and store the text embedding layer."""
        self._text_embedding_layer = None
        self._text_len = 512  # Default value
        
        # Search for text_embedding layer in various locations
        if hasattr(model, 'text_embedding'):
            self._text_embedding_layer = model.text_embedding
            if hasattr(model, 'text_len'):
                self._text_len = model.text_len
            verbose_print(f"Found text_embedding layer (text_len={self._text_len})")
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'text_embedding'):
            self._text_embedding_layer = model.transformer.text_embedding
            if hasattr(model.transformer, 'text_len'):
                self._text_len = model.transformer.text_len
            verbose_print(f"Found text_embedding layer in transformer (text_len={self._text_len})")
    
    def _create_patched_forward(self, block_idx, original_forward, injection_strength, injection_embeds):
        """Create a patched forward function for a transformer block."""
        patcher_instance = self
        
        @torch.compiler.disable
        def patched_forward(x, *args, **kwargs):
            # Always print for first few calls to verify
            if not hasattr(patched_forward, '_call_count'):
                patched_forward._call_count = 0
            patched_forward._call_count += 1
            
            # Force print for debugging - use plain print to ensure visibility
            if patched_forward._call_count <= 3:
                print(f"\n[ACTIVATION PATCH] Block {block_idx} forward called!")
                print(f"  Call #{patched_forward._call_count}")
                print(f"  x.shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
                print(f"  args count: {len(args)}")
                if len(args) > 5:
                    print(f"  context (arg 5) shape: {args[5].shape if hasattr(args[5], 'shape') else type(args[5])}")
                print(f"  kwargs keys: {list(kwargs.keys())}")
                print(f"  injection_embeds is None: {injection_embeds is None}")
                sys.stdout.flush()  # Force flush output
            
            if patched_forward._call_count == 1:
                debug_print(f"Block {block_idx}: Forward method called (first time)")
                debug_print(f"  x shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
                debug_print(f"  args count: {len(args)}")
                debug_print(f"  kwargs keys: {list(kwargs.keys())}")
            
            # WanAttentionBlock forward signature:
            # forward(self, x, e, seq_lens, grid_sizes, freqs, context, current_step, ...)
            # From the output, we can see context is passed as a kwarg
            context = None
            
            # Check if context is in kwargs (this is the typical case)
            if 'context' in kwargs:
                context = kwargs['context']
                if patched_forward._call_count <= 3:
                    debug_print(f"Block {block_idx}: Found context in kwargs")
                    if hasattr(context, 'shape'):
                        debug_print(f"Block {block_idx}: Context shape: {context.shape}")
            
            # Only process if we have both context and injection embeddings
            if context is not None and injection_embeds is not None:
                try:
                    # Process injection
                    blended_context = patcher_instance._process_injection(
                        block_idx, context, injection_embeds, injection_strength
                    )
                    
                    if blended_context is not None:
                        # Update context in kwargs
                        kwargs['context'] = blended_context
                        
                        # Call original forward with modified context
                        output = original_forward(x, *args, **kwargs)
                        
                        # Log first successful injection
                        if not hasattr(patched_forward, '_logged'):
                            debug_print(f"Block {block_idx}: Successfully applied injection")
                            patched_forward._logged = True
                        
                        return output
                        
                except Exception as e:
                    debug_print(f"Block {block_idx}: Error during injection: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Fall back to original forward
            return original_forward(x, *args, **kwargs)
        
        return patched_forward
    
    def _process_injection(self, block_idx, context, injection_embeds, injection_strength):
        """Process the injection and blend with original context."""
        # Log initial state
        debug_print(f"\n=== Block {block_idx} Injection Processing ===")
        verbose_print(f"Context type: {type(context)}, Injection type: {type(injection_embeds)}")
        
        # Extract tensors from dicts if needed
        if isinstance(context, dict):
            context = context.get('prompt_embeds', context)
            verbose_print("Extracted context from dict")
        if isinstance(injection_embeds, dict):
            injection_embeds = injection_embeds.get('prompt_embeds', injection_embeds)
            verbose_print("Extracted injection_embeds from dict")
        
        # Handle list format
        if isinstance(context, list):
            context = context[0] if len(context) > 0 else None
            verbose_print("Extracted context from list")
        if isinstance(injection_embeds, list):
            injection_embeds = injection_embeds[0] if len(injection_embeds) > 0 else None
            verbose_print("Extracted injection_embeds from list")
        
        if not torch.is_tensor(context) or not torch.is_tensor(injection_embeds):
            debug_print(f"Block {block_idx}: Invalid tensors - context is tensor: {torch.is_tensor(context)}, injection is tensor: {torch.is_tensor(injection_embeds)}")
            return None
        
        # Log tensor info
        debug_print(f"Block {block_idx} shapes - Context: {context.shape}, Injection: {injection_embeds.shape}")
        debug_print(f"Block {block_idx} devices - Context: {context.device}, Injection: {injection_embeds.device}")
        debug_print(f"Block {block_idx} dtypes - Context: {context.dtype}, Injection: {injection_embeds.dtype}")
        
        # Ensure same device and dtype
        if injection_embeds.device != context.device or injection_embeds.dtype != context.dtype:
            injection_embeds = injection_embeds.to(device=context.device, dtype=context.dtype)
            verbose_print(f"Moved injection to device={context.device}, dtype={context.dtype}")
        
        # Handle dimension mismatch
        if context.shape[-1] != injection_embeds.shape[-1]:
            debug_print(f"Block {block_idx}: Dimension mismatch {injection_embeds.shape[-1]} vs {context.shape[-1]}, projecting...")
            injection_embeds = self._project_embeddings(injection_embeds, context)
            if injection_embeds is None:
                debug_print(f"Block {block_idx}: Projection failed!")
                return None
            debug_print(f"Block {block_idx}: After projection shape: {injection_embeds.shape}")
        
        # Align shapes
        injection_embeds = self._align_shapes(injection_embeds, context)
        
        # Blend contexts
        blended = (1 - injection_strength) * context + injection_strength * injection_embeds
        
        # Calculate and log the actual change
        diff = (blended - context).abs()
        diff_percent = (diff > 0.01).float().mean().item() * 100
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        debug_print(f"Block {block_idx} Blending Results:")
        debug_print(f"  - Strength applied: {injection_strength}")
        debug_print(f"  - Mean difference: {mean_diff:.6f}")
        debug_print(f"  - Max difference: {max_diff:.6f}")
        debug_print(f"  - Percent changed: {diff_percent:.1f}%")
        
        # Enhanced debugging at injection point
        if VERBOSE:
            debug_injection_point(block_idx, context, injection_embeds, injection_strength)
        
        # Log statistics for first block only
        if block_idx == 0 or (block_idx % 10 == 0):
            self._log_blend_statistics(context, injection_embeds, blended, injection_strength)
        
        return blended
    
    def _project_embeddings(self, injection_embeds, context):
        """Project embeddings through text_embedding layer if needed."""
        if self._text_embedding_layer is None:
            verbose_print(f"No text_embedding layer available for projection")
            return None
        
        try:
            # Pad to text_len if needed
            if injection_embeds.shape[1] < self._text_len:
                padding = torch.zeros(
                    injection_embeds.shape[0], 
                    self._text_len - injection_embeds.shape[1], 
                    injection_embeds.shape[2],
                    device=injection_embeds.device, 
                    dtype=injection_embeds.dtype
                )
                injection_embeds = torch.cat([injection_embeds, padding], dim=1)
            elif injection_embeds.shape[1] > self._text_len:
                injection_embeds = injection_embeds[:, :self._text_len, :]
            
            # Get text embedding layer's device and dtype
            text_emb_params = next(self._text_embedding_layer.parameters())
            
            # Project through text_embedding layer
            injection_embeds = injection_embeds.to(device=text_emb_params.device, dtype=text_emb_params.dtype)
            projected = self._text_embedding_layer(injection_embeds)
            
            # Move back to context's device/dtype
            return projected.to(device=context.device, dtype=context.dtype)
            
        except Exception as e:
            verbose_print(f"Failed to project embeddings: {e}")
            return None
    
    def _align_shapes(self, injection_embeds, context):
        """Align injection embeddings shape to match context."""
        # Align sequence length
        if context.shape[1] != injection_embeds.shape[1]:
            if injection_embeds.shape[1] > context.shape[1]:
                injection_embeds = injection_embeds[:, :context.shape[1], :]
            else:
                padding = torch.zeros(
                    injection_embeds.shape[0], 
                    context.shape[1] - injection_embeds.shape[1], 
                    injection_embeds.shape[2],
                    device=injection_embeds.device, 
                    dtype=injection_embeds.dtype
                )
                injection_embeds = torch.cat([injection_embeds, padding], dim=1)
        
        # Align batch dimension
        if context.shape[0] != injection_embeds.shape[0]:
            injection_embeds = injection_embeds.expand(context.shape[0], -1, -1)
        
        return injection_embeds
    
    def _log_blend_statistics(self, context, injection_embeds, blended, injection_strength):
        """Log statistics about the blending operation."""
        with DebugBlock("blend_statistics"):
            context_norm = torch.norm(context).item()
            injection_norm = torch.norm(injection_embeds).item()
            blended_norm = torch.norm(blended).item()
            diff_norm = torch.norm(blended - context).item()
            
            debug_print(f"Injection statistics:")
            debug_print(f"  - Strength: {injection_strength}")
            debug_print(f"  - Context norm: {context_norm:.3f}")
            debug_print(f"  - Injection norm: {injection_norm:.3f}")
            debug_print(f"  - Blended norm: {blended_norm:.3f}")
            debug_print(f"  - Change: {(diff_norm/context_norm)*100:.1f}%")
            
            # More detailed tensor analysis
            debug_tensor(context, "context", detailed=True)
            debug_tensor(injection_embeds, "injection", detailed=True)
            debug_tensor(blended, "blended", detailed=True)
        
        # Analyze sequence composition if large
        if context.shape[1] > 1000 and VERBOSE:
            seq_len = context.shape[1]
            text_tokens = 512
            video_tokens = seq_len - text_tokens
            debug_print(f"\nSequence analysis:")
            debug_print(f"  - Total length: {seq_len}")
            debug_print(f"  - Text tokens: {text_tokens}")
            debug_print(f"  - Video tokens: {video_tokens}")
            
            # Try to factor video tokens
            for frames in [16, 24, 32, 41, 48, 64]:
                if video_tokens % frames == 0:
                    tokens_per_frame = video_tokens // frames
                    sqrt_val = math.sqrt(tokens_per_frame)
                    if sqrt_val == int(sqrt_val):
                        debug_print(f"  - {frames} frames × {int(sqrt_val)}×{int(sqrt_val)} patches")


class ContextPreprocessor:
    """Handles preprocessing of injection embeddings to match transformer expectations."""
    
    @staticmethod
    def prepare_injection_context(injection_embeds, reference_context=None):
        """
        Prepare injection embeddings to match the format expected by transformer blocks.
        
        Args:
            injection_embeds: Raw injection embeddings (from text encoder)
            reference_context: Optional reference for shape matching
        
        Returns:
            Processed context ready for injection
        """
        # Extract tensor from dict if needed
        if isinstance(injection_embeds, dict):
            injection_tensor = injection_embeds.get('prompt_embeds')
            if isinstance(injection_tensor, list):
                injection_tensor = injection_tensor[0] if len(injection_tensor) > 0 else None
        else:
            injection_tensor = injection_embeds
        
        if injection_tensor is None:
            return None
        
        # Ensure 2D tensor [seq_len, dim]
        if injection_tensor.dim() == 3 and injection_tensor.shape[0] == 1:
            injection_tensor = injection_tensor.squeeze(0)
        
        return injection_tensor


# Global patcher instance
_global_patcher = ActivationPatcher()

def get_patcher() -> ActivationPatcher:
    """Get the global activation patcher instance."""
    return _global_patcher