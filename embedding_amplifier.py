"""
Simple embedding amplifier to ensure embeddings are different enough for visible effects.
"""

import torch
from typing import Dict, Any
from .debug_utils import debug_print, verbose_print


class WanVideoEmbeddingAmplifier:
    """
    Amplifies the difference between two embeddings to ensure visible injection effects.
    Simple solution: if embeddings aren't different enough, make them more different.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_embeds": ("WANVIDEOTEXTEMBEDS",),
                "injection_embeds": ("WANVIDEOTEXTEMBEDS",),
                "target_difference": ("FLOAT", {
                    "default": 60.0, 
                    "min": 30.0, 
                    "max": 90.0, 
                    "step": 5.0,
                    "label": "Target Difference %"
                }),
                "amplification_mode": (["push_apart", "maximize_diff", "orthogonalize"], {
                    "default": "push_apart"
                }),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS", "STRING")
    RETURN_NAMES = ("amplified_injection", "info")
    FUNCTION = "amplify_difference"
    CATEGORY = "WanVideoWrapper/Tools"
    DESCRIPTION = "Makes injection embeddings more different from main embeddings for stronger effects"

    def amplify_difference(self, main_embeds, injection_embeds, 
                         target_difference=60.0, amplification_mode="push_apart"):
        
        # Extract tensors
        main_tensor = main_embeds.get('prompt_embeds')
        inj_tensor = injection_embeds.get('prompt_embeds')
        
        if isinstance(main_tensor, list):
            main_tensor = main_tensor[0]
        if isinstance(inj_tensor, list):
            inj_tensor = inj_tensor[0]
        
        if not torch.is_tensor(main_tensor) or not torch.is_tensor(inj_tensor):
            return (injection_embeds, "Error: Invalid embeddings")
        
        # Ensure same device
        if main_tensor.device != inj_tensor.device:
            inj_tensor = inj_tensor.to(main_tensor.device)
        
        # Ensure same shape
        if main_tensor.shape != inj_tensor.shape:
            # Simple truncation/padding
            if inj_tensor.shape[0] > main_tensor.shape[0]:
                inj_tensor = inj_tensor[:main_tensor.shape[0]]
            elif inj_tensor.shape[0] < main_tensor.shape[0]:
                padding = torch.zeros(
                    main_tensor.shape[0] - inj_tensor.shape[0],
                    inj_tensor.shape[1],
                    device=inj_tensor.device,
                    dtype=inj_tensor.dtype
                )
                inj_tensor = torch.cat([inj_tensor, padding], dim=0)
        
        # Calculate current difference
        diff = (main_tensor - inj_tensor).abs()
        current_diff_percent = (diff > 0.01).float().mean().item() * 100
        
        info_lines = []
        info_lines.append(f"Original difference: {current_diff_percent:.1f}%")
        
        # If already different enough, just return
        if current_diff_percent >= target_difference:
            info_lines.append(f"Already at target! No amplification needed.")
            return (injection_embeds, "\n".join(info_lines))
        
        # Amplify based on mode
        if amplification_mode == "push_apart":
            # Simple approach: push embeddings apart
            diff_vector = inj_tensor - main_tensor
            
            # Calculate how much to amplify
            if current_diff_percent > 0:
                amplification_factor = target_difference / current_diff_percent
            else:
                amplification_factor = 2.0
            
            # Apply amplification
            amplified_inj = main_tensor + diff_vector * amplification_factor
            
        elif amplification_mode == "maximize_diff":
            # More aggressive: maximize difference in high-variance dimensions
            diff_vector = inj_tensor - main_tensor
            
            # Find dimensions with highest variance in the difference
            dim_variance = diff_vector.var(dim=0)
            top_dims = torch.topk(dim_variance, k=int(diff_vector.shape[1] * 0.3)).indices
            
            # Amplify only top dimensions
            amplified_inj = inj_tensor.clone()
            amplification_factor = target_difference / max(current_diff_percent, 1.0)
            
            for dim in top_dims:
                amplified_inj[:, dim] += diff_vector[:, dim] * (amplification_factor - 1.0)
            
        else:  # orthogonalize
            # Make injection more orthogonal to main
            # Project out the component parallel to main
            main_norm = main_tensor / (main_tensor.norm(dim=-1, keepdim=True) + 1e-6)
            parallel_component = (inj_tensor * main_norm).sum(dim=-1, keepdim=True) * main_norm
            orthogonal_component = inj_tensor - parallel_component
            
            # Mix to achieve target difference
            mix_ratio = min(target_difference / 100.0, 0.9)
            amplified_inj = (1 - mix_ratio) * inj_tensor + mix_ratio * orthogonal_component
        
        # Normalize to preserve overall magnitude
        orig_norm = inj_tensor.norm(dim=-1, keepdim=True)
        amplified_norm = amplified_inj.norm(dim=-1, keepdim=True)
        amplified_inj = amplified_inj * (orig_norm / (amplified_norm + 1e-6))
        
        # Calculate final difference
        final_diff = (main_tensor - amplified_inj).abs()
        final_diff_percent = (final_diff > 0.01).float().mean().item() * 100
        
        info_lines.append(f"Amplified difference: {final_diff_percent:.1f}%")
        info_lines.append(f"Mode: {amplification_mode}")
        
        # Create output embeddings
        output_embeds = injection_embeds.copy()
        output_embeds['prompt_embeds'] = amplified_inj
        
        # Print to console
        print(f"\n[Embedding Amplifier]")
        for line in info_lines:
            print(f"  {line}")
        
        return (output_embeds, "\n".join(info_lines))


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVideoEmbeddingAmplifier": WanVideoEmbeddingAmplifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoEmbeddingAmplifier": "WanVideo Embedding Amplifier",
}