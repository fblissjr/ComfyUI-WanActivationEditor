"""
Alternative injection methods for WanVideo activation editing.
These approaches work at different levels of the generation pipeline.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Callable
import numpy as np


class LatentSpaceInjector:
    """
    Inject activations by manipulating latent space during denoising steps.
    Works by intercepting and modifying latents between denoising iterations.
    """
    
    def __init__(self):
        self.injection_schedule = {}
        self.original_callbacks = {}
    
    def create_step_callback(self, active_blocks: List[int], injection_strength: float, 
                           injection_latents: torch.Tensor) -> Callable:
        """
        Create a callback that modifies latents at specific denoising steps.
        
        This works by blending the trajectory of two different denoising paths.
        """
        def step_callback(step: int, latent: torch.Tensor, total_steps: int) -> torch.Tensor:
            # Calculate which "blocks" this step corresponds to
            # Map denoising steps to virtual block indices
            block_index = int((step / total_steps) * 40)
            
            if block_index in active_blocks:
                # Blend latents based on injection strength
                # This creates a hybrid denoising trajectory
                modified_latent = (1 - injection_strength) * latent + injection_strength * injection_latents
                print(f"[LatentInjector] Step {step}/{total_steps} (Block {block_index}): Injected with strength {injection_strength}")
                return modified_latent
            
            return latent
        
        return step_callback


class CrossAttentionHijacker:
    """
    Hijack cross-attention layers to inject alternative conditioning.
    Works by modifying attention weights during inference.
    """
    
    def __init__(self):
        self.attention_processors = {}
        self.injection_config = {}
    
    def create_attention_processor(self, block_idx: int, injection_strength: float,
                                 injection_context: torch.Tensor, original_context: torch.Tensor):
        """
        Create a custom attention processor that blends contexts.
        """
        class BlendedAttentionProcessor:
            def __init__(self, strength, inj_ctx, orig_ctx):
                self.strength = strength
                self.inj_ctx = inj_ctx
                self.orig_ctx = orig_ctx
            
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, **kwargs):
                # If this is cross-attention and we have contexts
                if encoder_hidden_states is not None:
                    # Blend the contexts
                    blended_context = (1 - self.strength) * encoder_hidden_states + self.strength * self.inj_ctx
                    # Call original attention with blended context
                    return attn._orig_forward(hidden_states, blended_context, **kwargs)
                else:
                    # Self-attention, pass through
                    return attn._orig_forward(hidden_states, encoder_hidden_states, **kwargs)
        
        return BlendedAttentionProcessor(injection_strength, injection_context, original_context)


class GuidanceModulator:
    """
    Modulate classifier-free guidance to achieve injection effects.
    Works by creating custom guidance that blends different conditioning paths.
    """
    
    @staticmethod
    def create_mixed_guidance(cond_pred: torch.Tensor, uncond_pred: torch.Tensor, 
                            alt_cond_pred: torch.Tensor, cfg_scale: float,
                            injection_strength: float, active_blocks: List[int],
                            current_step: int, total_steps: int) -> torch.Tensor:
        """
        Create custom guidance that blends multiple conditioning paths.
        
        Standard CFG: pred = uncond + cfg_scale * (cond - uncond)
        Mixed CFG: pred = uncond + cfg_scale * ((1-strength)*cond + strength*alt_cond - uncond)
        """
        # Map current step to block index
        block_index = int((current_step / total_steps) * 40)
        
        if block_index in active_blocks:
            # Blend conditional predictions
            mixed_cond = (1 - injection_strength) * cond_pred + injection_strength * alt_cond_pred
            # Apply CFG with mixed conditioning
            pred = uncond_pred + cfg_scale * (mixed_cond - uncond_pred)
            print(f"[GuidanceModulator] Step {current_step}/{total_steps}: Mixed guidance applied")
        else:
            # Standard CFG
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        
        return pred


class NoiseScheduleManipulator:
    """
    Manipulate the noise schedule to create injection effects.
    Works by using different noise patterns for different semantic regions.
    """
    
    @staticmethod
    def create_hybrid_noise(shape: tuple, active_blocks: List[int], 
                          main_seed: int, injection_seed: int,
                          injection_strength: float) -> torch.Tensor:
        """
        Create noise that encodes different patterns in different regions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate two different noise patterns
        main_gen = torch.Generator(device=device).manual_seed(main_seed)
        inject_gen = torch.Generator(device=device).manual_seed(injection_seed)
        
        main_noise = torch.randn(shape, generator=main_gen, device=device)
        inject_noise = torch.randn(shape, generator=inject_gen, device=device)
        
        # Create spatial mask based on active blocks
        # This is a simplified example - real implementation would be more sophisticated
        if len(active_blocks) > 0:
            # Map blocks to spatial regions (simplified)
            mask = torch.zeros(shape, device=device)
            
            # Late blocks affect high-level features (larger spatial regions)
            if max(active_blocks) > 30:  # Late blocks
                mask[:, :, :, :] = injection_strength
            elif max(active_blocks) > 20:  # Mid blocks
                # Checkerboard pattern for mid-level features
                for i in range(0, shape[2], 2):
                    for j in range(0, shape[3], 2):
                        mask[:, :, i, j] = injection_strength
            else:  # Early blocks
                # Fine-grained pattern for low-level features
                mask[:, :, ::4, ::4] = injection_strength
            
            # Blend noises
            hybrid_noise = (1 - mask) * main_noise + mask * inject_noise
        else:
            hybrid_noise = main_noise
        
        return hybrid_noise


class FeatureMapBlender:
    """
    Blend feature maps at specific layers during forward pass.
    Works by caching and blending intermediate activations.
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.blend_configs = {}
    
    def cache_features(self, name: str, features: torch.Tensor):
        """Cache intermediate features during forward pass."""
        self.feature_cache[name] = features.clone()
    
    def blend_features(self, name: str, features: torch.Tensor, 
                      injection_features: torch.Tensor, strength: float) -> torch.Tensor:
        """Blend current features with injected features."""
        if injection_features is not None:
            # Ensure shape compatibility
            if features.shape == injection_features.shape:
                blended = (1 - strength) * features + strength * injection_features
                print(f"[FeatureBlender] Blended features at {name} with strength {strength}")
                return blended
            else:
                print(f"[FeatureBlender] Shape mismatch at {name}, skipping blend")
        
        return features


class SamplerInterceptor:
    """
    Intercept and modify the sampling process itself.
    Works by wrapping the sampler with custom logic.
    """
    
    @staticmethod
    def create_wrapped_sampler(original_sampler, injection_config: Dict[str, Any]):
        """
        Create a wrapped sampler that injects activations during sampling.
        """
        class WrappedSampler:
            def __init__(self, sampler, config):
                self.sampler = sampler
                self.config = config
                self.step_count = 0
            
            def __call__(self, *args, **kwargs):
                # Extract relevant info
                model = args[0] if args else kwargs.get('model')
                latents = args[1] if len(args) > 1 else kwargs.get('latents')
                
                # Check if we should inject at this step
                active_blocks = self.config.get('active_blocks', [])
                injection_strength = self.config.get('injection_strength', 0.5)
                
                # Map step to block range
                total_steps = self.config.get('total_steps', 20)
                block_index = int((self.step_count / total_steps) * 40)
                
                if block_index in active_blocks:
                    print(f"[SamplerInterceptor] Injecting at step {self.step_count} (block {block_index})")
                    # Modify kwargs to influence sampling
                    if 'conditioning' in kwargs:
                        # Blend conditioning
                        orig_cond = kwargs['conditioning']
                        inj_cond = self.config.get('injection_conditioning')
                        if inj_cond is not None:
                            kwargs['conditioning'] = self._blend_conditioning(
                                orig_cond, inj_cond, injection_strength
                            )
                
                self.step_count += 1
                
                # Call original sampler
                return self.sampler(*args, **kwargs)
            
            def _blend_conditioning(self, orig, inj, strength):
                """Blend two conditioning tensors."""
                if isinstance(orig, torch.Tensor) and isinstance(inj, torch.Tensor):
                    return (1 - strength) * orig + strength * inj
                return orig
            
            def __getattr__(self, name):
                # Delegate attribute access to original sampler
                return getattr(self.sampler, name)
        
        return WrappedSampler(original_sampler, injection_config)


# Export all injection methods
__all__ = [
    'LatentSpaceInjector',
    'CrossAttentionHijacker', 
    'GuidanceModulator',
    'NoiseScheduleManipulator',
    'FeatureMapBlender',
    'SamplerInterceptor'
]