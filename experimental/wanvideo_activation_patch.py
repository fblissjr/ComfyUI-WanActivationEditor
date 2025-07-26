"""
WanVideo Activation Editor Patch

This file shows how to modify WanVideoWrapper to support activation editing.
Apply this patch to WanVideoWrapper/wanvideo/modules/model.py

The patch adds support for reading activation_editor configuration from transformer_options
and applying prompt injection at the block level during the forward pass.

To apply:
1. Locate the block processing loop in WanVideoWrapper/wanvideo/modules/model.py (around line 1795)
2. Add the activation editor code before the block forward call
3. The activation editor will blend contexts for specified blocks
"""

# Add this import at the top of model.py
# import torch

# Add this code in the forward method, right before the block loop (around line 1790):
"""
# Activation Editor Support
activation_editor = None
if hasattr(self, 'transformer_options') and self.transformer_options:
    activation_editor = self.transformer_options.get('activation_editor', None)
elif kwargs.get('transformer_options'):
    activation_editor = kwargs['transformer_options'].get('activation_editor', None)
"""

# Replace the block loop (around line 1795) with this modified version:
"""
for b, block in enumerate(self.blocks):
    # Skip blocks based on various conditions
    if b > self.blocks_to_swap and self.blocks_to_swap >= 0:
        if b in self.slg_blocks and is_uncond:
            if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                continue
    if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
        block.to(self.main_device)
    
    # Activation Editor: Check if we should inject alternative context
    if activation_editor and activation_editor.get('enabled', False):
        block_activations = activation_editor.get('block_activations', [])
        if b < len(block_activations) and block_activations[b] == 1:
            # This block should have activation injection
            injection_strength = activation_editor.get('injection_strength', 0.5)
            injection_embeds = activation_editor.get('injection_embeds')
            
            if injection_embeds is not None:
                # Blend the contexts
                original_context = kwargs.get('context', context)
                blended_context = (1 - injection_strength) * original_context + injection_strength * injection_embeds
                
                # Create modified kwargs with blended context
                modified_kwargs = kwargs.copy()
                modified_kwargs['context'] = blended_context
                
                # Run block with blended context
                x = block(x, **modified_kwargs)
            else:
                # No injection embeds, run normally
                x = block(x, **kwargs)
        else:
            # Normal block execution
            x = block(x, **kwargs)
    else:
        # No activation editor, normal execution
        x = block(x, **kwargs)
    
    # Handle controlnet additions (keep existing code)
    if pdc_controlnet_states is not None and b < len(pdc_controlnet_states):
        x[:, :x_len] += pdc_controlnet_states[b].to(x) * pcd_data["controlnet_weight"]
    if (controlnet is not None) and (b % controlnet["controlnet_stride"] == 0) and (b // controlnet["controlnet_stride"] < len(controlnet["controlnet_states"])):
        x[:, :x_len] += controlnet["controlnet_states"][b // controlnet["controlnet_stride"]].to(x) * controlnet["controlnet_weight"]
    
    if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
        block.to(self.offload_device, non_blocking=self.use_non_blocking)
"""

# Alternative minimal patch if you prefer not to modify the core loop:
"""
# Add this method to the WanVideoTransformer class:
def _get_block_context(self, block_idx, kwargs, transformer_options=None):
    '''Get context for a specific block, with activation editor support'''
    context = kwargs.get('context')
    
    # Check for activation editor
    if transformer_options:
        activation_editor = transformer_options.get('activation_editor')
        if activation_editor and activation_editor.get('enabled', False):
            block_activations = activation_editor.get('block_activations', [])
            if block_idx < len(block_activations) and block_activations[block_idx] == 1:
                injection_strength = activation_editor.get('injection_strength', 0.5)
                injection_embeds = activation_editor.get('injection_embeds')
                if injection_embeds is not None:
                    # Blend contexts
                    context = (1 - injection_strength) * context + injection_strength * injection_embeds
    
    return context

# Then in the block loop, replace:
# x = block(x, **kwargs)
# With:
# block_kwargs = kwargs.copy()
# block_kwargs['context'] = self._get_block_context(b, kwargs, self.transformer_options if hasattr(self, 'transformer_options') else None)
# x = block(x, **block_kwargs)
"""