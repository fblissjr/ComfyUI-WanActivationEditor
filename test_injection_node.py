"""
Test node to verify injection is actually modifying generation.
Compares embeddings/hidden states before and after injection.
"""

import torch
from .debug_utils import debug_print, verbose_print, DEBUG, VERBOSE


class WanVideoInjectionTester:
    """
    Test node that verifies injection is working by comparing states.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "show_details": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("test_results",)
    FUNCTION = "test_injection"
    CATEGORY = "WanVideoWrapper/Debug"
    DESCRIPTION = "Test injection effectiveness by analyzing model state"

    def test_injection(self, model, text_embeds, show_details=True):
        lines = []
        lines.append("=== WanVideo Injection Test Results ===")
        
        # Check if model has activation configuration
        has_activation = False
        activation_config = None
        
        if 'transformer_options' in model.model_options:
            if 'wan_activation_editor' in model.model_options['transformer_options']:
                has_activation = True
                activation_config = model.model_options['transformer_options']['wan_activation_editor']
        
        lines.append(f"\nActivation Editor Configured: {has_activation}")
        
        if has_activation:
            lines.append(f"Active blocks: {activation_config.get('active_blocks', [])}")
            lines.append(f"Injection strength: {activation_config.get('injection_strength', 0)}")
            lines.append(f"Runtime patched: {activation_config.get('runtime_patched', False)}")
            
            # Check if injection embeddings are present
            if 'wan_activation_editor' in text_embeds:
                inj_info = text_embeds['wan_activation_editor']
                has_injection = inj_info.get('injection_embeds') is not None
                lines.append(f"Has injection embeddings: {has_injection}")
                
                if has_injection and show_details:
                    # Compare main and injection embeddings
                    main_embeds = text_embeds.get('prompt_embeds')
                    inj_embeds = inj_info.get('injection_embeds')
                    
                    if main_embeds is not None and inj_embeds is not None:
                        # Extract tensors
                        if isinstance(main_embeds, list):
                            main_embeds = main_embeds[0]
                        if isinstance(inj_embeds, list):
                            inj_embeds = inj_embeds[0]
                        
                        if torch.is_tensor(main_embeds) and torch.is_tensor(inj_embeds):
                            # Calculate difference
                            if main_embeds.device != inj_embeds.device:
                                inj_embeds = inj_embeds.to(main_embeds.device)
                            
                            diff = (main_embeds - inj_embeds).abs()
                            diff_percent = (diff > 0.01).float().mean().item() * 100
                            
                            lines.append(f"\n[Embedding Analysis]")
                            lines.append(f"Main shape: {main_embeds.shape}")
                            lines.append(f"Injection shape: {inj_embeds.shape}")
                            lines.append(f"Difference: {diff_percent:.1f}%")
                            
                            if diff_percent < 30:
                                lines.append("⚠️ WARNING: Low difference - prompts too similar!")
                            elif diff_percent < 50:
                                lines.append("⚠️ CAUTION: Moderate difference - effects may be subtle")
                            else:
                                lines.append("✓ Good difference - should see clear effects")
                            
                            # Analyze which dimensions differ most
                            if show_details and main_embeds.shape == inj_embeds.shape:
                                dim_diffs = diff.mean(dim=0)
                                if dim_diffs.dim() > 1:
                                    dim_diffs = dim_diffs.mean(dim=0)
                                
                                top_k = 10
                                top_dims = torch.topk(dim_diffs, min(top_k, dim_diffs.shape[0]))
                                
                                lines.append(f"\nTop {top_k} differing dimensions:")
                                for i, (val, idx) in enumerate(zip(top_dims.values, top_dims.indices)):
                                    lines.append(f"  Dim {idx.item()}: {val.item():.4f}")
                        else:
                            lines.append("\nCould not analyze embeddings - not tensors")
                    else:
                        lines.append("\nMissing embedding data for analysis")
            else:
                lines.append("No injection embeddings in text_embeds")
        else:
            lines.append("\nNo activation configuration found.")
            lines.append("Make sure to connect through WanVideoActivationEditor first.")
        
        # Print to console
        print("\n".join(lines))
        
        return ("\n".join(lines),)


# Add to node mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoInjectionTester": WanVideoInjectionTester,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoInjectionTester": "WanVideo Injection Tester",
}