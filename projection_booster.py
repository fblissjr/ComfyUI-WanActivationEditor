"""
Projection Booster - Works with already-projected embeddings to restore differences.
This is a simpler solution than full latent encoding.
"""

import torch
from typing import Dict, Any
from .debug_utils import debug_print, verbose_print


class WanVideoProjectionBooster:
    """
    Boosts the difference between already-projected embeddings (5120-dim).
    This works AFTER the text_embedding layer has already projected them.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "injection_embeds": ("WANVIDEOTEXTEMBEDS",),
                "boost_factor": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "label": "Boost Factor"
                }),
                "preserve_norm": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("model", "text_embeds", "info")
    FUNCTION = "boost_projection"
    CATEGORY = "WanVideoWrapper/Tools"
    DESCRIPTION = "Boosts embedding differences AFTER projection to 5120-dim"

    def boost_projection(self, model, text_embeds, injection_embeds, 
                        boost_factor=10.0, preserve_norm=True):
        """
        This node works by intercepting the embeddings AFTER they've been
        projected to 5120-dim and boosting their differences.
        """
        
        info_lines = []
        info_lines.append(f"Boost factor: {boost_factor}x")
        
        # Clone model
        m = model.clone()
        
        # We need to patch the model to boost embeddings after projection
        # This is done by adding a special flag that the activation patcher will use
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}
        
        # Extract the projected embeddings if they exist
        if hasattr(m, '_projected_context') and hasattr(m, '_projected_injection'):
            # We have access to already-projected embeddings
            main_proj = m._projected_context
            inj_proj = m._projected_injection
            
            # Calculate current difference
            diff = (main_proj - inj_proj).abs()
            current_diff = (diff > 0.01).float().mean().item() * 100
            info_lines.append(f"Current projected difference: {current_diff:.1f}%")
            
            # Boost the difference
            diff_vector = inj_proj - main_proj
            boosted_inj = main_proj + diff_vector * boost_factor
            
            # Preserve norm if requested
            if preserve_norm:
                orig_norm = inj_proj.norm(dim=-1, keepdim=True)
                boosted_norm = boosted_inj.norm(dim=-1, keepdim=True)
                boosted_inj = boosted_inj * (orig_norm / (boosted_norm + 1e-6))
            
            # Calculate new difference
            new_diff = (main_proj - boosted_inj).abs()
            new_diff_percent = (new_diff > 0.01).float().mean().item() * 100
            info_lines.append(f"Boosted difference: {new_diff_percent:.1f}%")
            
            # Store boosted embeddings
            m._projected_injection_boosted = boosted_inj
        
        # Add boost configuration
        m.model_options['transformer_options']['projection_boost'] = {
            'enabled': True,
            'boost_factor': boost_factor,
            'preserve_norm': preserve_norm
        }
        
        # Also store in text_embeds for compatibility
        modified_embeds = text_embeds.copy()
        if 'wan_activation_editor' in modified_embeds:
            modified_embeds['wan_activation_editor']['boost_factor'] = boost_factor
        
        # Print info
        print(f"\n[Projection Booster]")
        for line in info_lines:
            print(f"  {line}")
        
        return (m, modified_embeds, "\n".join(info_lines))


class WanVideoDirectInjector:
    """
    Directly modifies the context tensor to ensure differences are preserved.
    This is the most aggressive approach.
    """
    
    def __init__(self):
        self.patcher = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "injection_prompt": ("STRING", {"multiline": True}),
                "difference_target": ("FLOAT", {
                    "default": 50.0,
                    "min": 10.0,
                    "max": 90.0,
                    "step": 5.0,
                    "label": "Target Difference %"
                }),
                "injection_mode": (["additive", "replace", "blend"], {"default": "blend"}),
                "block_activations": ("STRING", {"default": "0" * 40}),
            },
            "optional": {
                "injection_embeds": ("WANVIDEOTEXTEMBEDS",),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "text_embeds")
    FUNCTION = "direct_inject"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "Directly modifies context to ensure visible differences"

    def direct_inject(self, model, text_embeds, injection_prompt, 
                     difference_target=50.0, injection_mode="blend",
                     block_activations="0"*40, injection_embeds=None):
        """
        Instead of relying on the model's projection, we directly create
        context tensors with the desired difference level.
        """
        
        # Parse blocks
        try:
            activations = [int(bit) for bit in block_activations]
            active_blocks = [i for i, a in enumerate(activations) if a == 1]
        except ValueError:
            raise ValueError("block_activations must be a string of 0s and 1s")
        
        if not active_blocks:
            print("[Direct Injector] No blocks activated")
            return (model, text_embeds)
        
        # Clone model
        m = model.clone()
        
        # Configure direct injection
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}
        
        m.model_options['transformer_options']['direct_injection'] = {
            'active_blocks': active_blocks,
            'injection_prompt': injection_prompt,
            'difference_target': difference_target,
            'injection_mode': injection_mode,
            'enabled': True
        }
        
        # Create modified embeddings
        modified_embeds = text_embeds.copy()
        modified_embeds['direct_injection'] = {
            'prompt': injection_prompt,
            'target_difference': difference_target,
            'mode': injection_mode
        }
        
        print(f"\n[Direct Injector]")
        print(f"  Mode: {injection_mode}")
        print(f"  Target difference: {difference_target}%")
        print(f"  Active blocks: {len(active_blocks)}")
        
        return (m, modified_embeds)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVideoProjectionBooster": WanVideoProjectionBooster,
    "WanVideoDirectInjector": WanVideoDirectInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoProjectionBooster": "WanVideo Projection Booster",
    "WanVideoDirectInjector": "WanVideo Direct Injector",
}