"""
Latent encoder that captures model's internal representations after initial processing.
This solves the dimension mismatch and projection issues.
"""

import torch
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
from .debug_utils import debug_print, verbose_print, debug_tensor
from .database import EmbeddingDatabase


class WanVideoLatentEncoder:
    """
    Captures latent representations from the model after initial block processing.
    This gives us embeddings in the model's native 5120-dim space that actually
    preserve differences.
    """
    
    def __init__(self):
        self.db = EmbeddingDatabase()
        self._cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "capture_after_blocks": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "label": "Capture After N Blocks"
                }),
                "use_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT_EMBEDS", "STRING")
    RETURN_NAMES = ("latent_embeds", "info")
    FUNCTION = "encode_to_latent"
    CATEGORY = "WanVideoWrapper/Advanced"
    DESCRIPTION = "Captures model's internal latent representations after initial processing"

    def encode_to_latent(self, model, text_embeds, capture_after_blocks=3, use_cache=True):
        """
        Run text embeddings through the first N transformer blocks to get
        proper latent representations in the model's 5120-dim space.
        """
        
        # Generate cache key
        cache_key = self._generate_cache_key(text_embeds, capture_after_blocks)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            debug_print(f"Found cached latent for key {cache_key[:8]}...")
            return self._cache[cache_key]
        
        info_lines = []
        info_lines.append(f"Encoding to latent space (after {capture_after_blocks} blocks)")
        
        try:
            # Access the actual model
            actual_model = self._find_model(model)
            if actual_model is None:
                return (None, "Error: Could not access model")
            
            # Find transformer blocks
            blocks = self._find_transformer_blocks(actual_model)
            if blocks is None or len(blocks) < capture_after_blocks:
                return (None, f"Error: Model has {len(blocks) if blocks else 0} blocks, need {capture_after_blocks}")
            
            # Get text embedding layer for initial projection
            text_embedding_layer = self._find_text_embedding_layer(actual_model)
            if text_embedding_layer is None:
                return (None, "Error: Could not find text_embedding layer")
            
            # Extract raw embeddings
            raw_embeds = text_embeds.get('prompt_embeds')
            if isinstance(raw_embeds, list):
                raw_embeds = raw_embeds[0]
            
            if not torch.is_tensor(raw_embeds):
                return (None, "Error: Invalid text embeddings")
            
            info_lines.append(f"Input shape: {raw_embeds.shape}")
            
            # Ensure correct format for text_embedding layer
            if raw_embeds.dim() == 2:
                raw_embeds = raw_embeds.unsqueeze(0)
            
            # Move to model device
            device = next(text_embedding_layer.parameters()).device
            dtype = next(text_embedding_layer.parameters()).dtype
            raw_embeds = raw_embeds.to(device=device, dtype=dtype)
            
            # Log device/dtype info
            info_lines.append(f"Device: {device}, Dtype: {dtype}")
            
            # Check if model uses quantization
            is_quantized = str(dtype).endswith('e4m3fn') or str(dtype).endswith('e5m2')
            if is_quantized:
                info_lines.append(f"Model is using FP8 quantization: {dtype}")
            
            # Pass through text_embedding layer
            with torch.no_grad():
                # Project to 5120-dim
                latent = text_embedding_layer(raw_embeds)
                info_lines.append(f"After projection: {latent.shape}")
                
                # Create a more complete set of kwargs that matches actual generation
                batch_size = 1
                seq_len = latent.shape[1]
                
                # Initialize hidden states - this represents the latent video tokens
                # Using random noise to simulate actual latents
                x = torch.randn(batch_size, seq_len * 32, 5120, device=device, dtype=dtype) * 0.1
                
                # Create realistic kwargs
                mock_kwargs = {
                    'context': latent,
                    'e': torch.zeros(batch_size, 5120, device=device, dtype=dtype),  # Timestep embedding
                    'seq_lens': [x.shape[1]],
                    'grid_sizes': [(8, 4)],  # Reasonable grid for video
                    'freqs': None,
                    'clip_embed': torch.zeros(batch_size, 768, device=device, dtype=dtype),  # CLIP embedding
                    'current_step': 0,
                    'last_step': 10,
                    'video_attention_split_steps': [],
                    'camera_embed': None,
                    'audio_proj': None,
                    'num_latent_frames': 32,
                    'enhance_enabled': False,
                    'audio_scale': 0.0,
                    'block_mask': None,
                    'nag_params': None,
                    'nag_context': None,
                    'is_uncond': False,
                    'multitalk_audio_embedding': None,
                    'ref_target_masks': None,
                    'human_num': 0,
                    'inner_t': None,
                    'inner_c': None,
                    'cross_freqs': None,
                }
                
                # Store original context for comparison
                original_context = latent.clone()
                
                # Pass through first N blocks
                for i in range(capture_after_blocks):
                    block = blocks[i]
                    debug_print(f"Processing through block {i}")
                    
                    # Call block forward - this will modify x and potentially context
                    x = block(x, **mock_kwargs)
                    
                    # Some blocks might return tuples
                    if isinstance(x, tuple):
                        x = x[0]
                
                info_lines.append(f"Hidden state shape after {capture_after_blocks} blocks: {x.shape}")
                
                # Extract a subset of the hidden states as our latent representation
                # We'll take the first seq_len tokens which should correspond to text influence
                latent_output = x[:, :seq_len, :].contiguous()
                
                # Measure how different it is from original
                if raw_embeds.shape == latent_output.shape:
                    diff = (raw_embeds - latent_output).abs()
                    diff_percent = (diff > 0.01).float().mean().item() * 100
                    info_lines.append(f"Latent difference from input: {diff_percent:.1f}%")
            
            # Store device and dtype info before moving to CPU
            original_device = latent_output.device
            original_dtype = latent_output.dtype
            
            # Move to CPU for storage
            latent_output = latent_output.cpu()
            
            # Create output structure with device/dtype info
            latent_embeds = {
                'latent': latent_output,
                'shape': list(latent_output.shape),
                'capture_blocks': capture_after_blocks,
                'source_prompt': text_embeds.get('prompt', 'unknown'),
                'device': str(original_device),
                'dtype': str(original_dtype),
                'is_quantized': str(original_dtype).endswith('e4m3fn') or str(original_dtype).endswith('e5m2')
            }
            
            # Cache if enabled
            if use_cache:
                self._cache[cache_key] = (latent_embeds, "\n".join(info_lines))
            
            # Store in database
            self.db.store_latent(latent_embeds, text_embeds.get('prompt', ''))
            
            # Print info
            print(f"\n[Latent Encoder]")
            for line in info_lines:
                print(f"  {line}")
            
            return (latent_embeds, "\n".join(info_lines))
            
        except Exception as e:
            error_msg = f"Error during latent encoding: {str(e)}"
            debug_print(error_msg)
            import traceback
            traceback.print_exc()
            return (None, error_msg)
    
    def _generate_cache_key(self, text_embeds, blocks):
        """Generate a cache key for the embeddings."""
        if 'prompt' in text_embeds:
            prompt_hash = hashlib.md5(text_embeds['prompt'].encode()).hexdigest()[:8]
        else:
            embeds = text_embeds.get('prompt_embeds')
            if isinstance(embeds, list):
                embeds = embeds[0]
            if torch.is_tensor(embeds):
                # Use tensor values for hash
                prompt_hash = hashlib.md5(embeds.cpu().numpy().tobytes()).hexdigest()[:8]
            else:
                prompt_hash = "unknown"
        
        return f"{prompt_hash}_blocks{blocks}"
    
    def _find_model(self, model_patcher):
        """Find the actual model inside the ModelPatcher."""
        if hasattr(model_patcher, 'model'):
            model = model_patcher.model
            if hasattr(model, 'diffusion_model'):
                return model.diffusion_model
            elif hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                return model.model.diffusion_model
            return model
        return None
    
    def _find_transformer_blocks(self, model):
        """Find transformer blocks in the model."""
        if hasattr(model, 'blocks'):
            return model.blocks
        elif hasattr(model, 'transformer_blocks'):
            return model.transformer_blocks
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            return model.transformer.blocks
        return None
    
    def _find_text_embedding_layer(self, model):
        """Find the text embedding layer."""
        if hasattr(model, 'text_embedding'):
            return model.text_embedding
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'text_embedding'):
            return model.transformer.text_embedding
        return None


class WanVideoLatentInjector:
    """
    Uses latent embeddings for injection instead of raw text embeddings.
    This should produce much stronger effects.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "main_latent": ("LATENT_EMBEDS",),
                "injection_latent": ("LATENT_EMBEDS",),
                "injection_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "block_activations": ("STRING", {"default": "0" * 40}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject_latents"
    CATEGORY = "WanVideoWrapper/Advanced"
    DESCRIPTION = "Injects latent representations for stronger activation editing"

    def inject_latents(self, model, main_latent, injection_latent, 
                      injection_strength, block_activations):
        """
        Configure model to use latent embeddings for injection.
        """
        
        # Parse block activations
        try:
            activations = [int(bit) for bit in block_activations]
            active_blocks = [i for i, a in enumerate(activations) if a == 1]
        except ValueError:
            raise ValueError("block_activations must be a string of 0s and 1s")
        
        if not active_blocks:
            print("[Latent Injector] No blocks activated")
            return (model,)
        
        # Extract latent tensors
        main_tensor = main_latent.get('latent')
        inj_tensor = injection_latent.get('latent')
        
        if main_tensor is None or inj_tensor is None:
            print("[Latent Injector] Invalid latent embeddings")
            return (model,)
        
        # Measure difference
        if main_tensor.shape == inj_tensor.shape:
            diff = (main_tensor - inj_tensor).abs()
            diff_percent = (diff > 0.01).float().mean().item() * 100
            print(f"\n[Latent Injector]")
            print(f"  Latent difference: {diff_percent:.1f}%")
            print(f"  Active blocks: {len(active_blocks)}")
            print(f"  Injection strength: {injection_strength}")
        
        # Clone model
        m = model.clone()
        
        # Configure for latent injection
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}
        
        # Get device/dtype info from latent embeddings
        device_str = main_latent.get('device', 'cuda:0')
        dtype_str = main_latent.get('dtype', 'torch.float16')
        
        m.model_options['transformer_options']['latent_injection'] = {
            'main_latent': main_tensor,
            'injection_latent': inj_tensor,
            'active_blocks': active_blocks,
            'injection_strength': injection_strength,
            'device': device_str,
            'dtype': dtype_str,
            'enabled': True
        }
        
        return (m,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVideoLatentEncoder": WanVideoLatentEncoder,
    "WanVideoLatentInjector": WanVideoLatentInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLatentEncoder": "WanVideo Latent Encoder",
    "WanVideoLatentInjector": "WanVideo Latent Injector",
}