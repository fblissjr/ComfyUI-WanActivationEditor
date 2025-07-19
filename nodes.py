import torch
import comfy.model_management as mm
from types import MethodType

class WanVideoActivationEditor:
    @classmethod
    def INPUT_TYPES(cls):
        default_activations = "0" * 40
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "injection_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "block_activations": ("STRING", {"default": default_activations, "multiline": False}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "injection_conditioning": ("WANVIDEOTEXTEMBEDS",),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "apply_activation_editing"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "Inject alternative conditioning into specific transformer blocks to experiment with how different blocks contribute to generation. Compatible with WanVideoWrapper's scheduling system."

    def apply_activation_editing(self, model, injection_strength, block_activations, start_percent, end_percent, injection_conditioning=None):
        model_clone = model.clone()
        
        # Parse block activations
        try:
            activations = [int(bit) for bit in block_activations]
        except ValueError:
            raise ValueError("`block_activations` must be a string of 0s and 1s (e.g., '00011100000...')")

        # Store injection parameters in transformer options
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        
        # Support multiple activation injections
        if 'activation_injections' not in model_clone.model_options['transformer_options']:
            model_clone.model_options['transformer_options']['activation_injections'] = []
        
        model_clone.model_options['transformer_options']['activation_injections'].append({
            'conditioning': injection_conditioning,
            'strength': injection_strength,
            'block_activations': activations,
            'start_percent': start_percent,
            'end_percent': end_percent
        })

        # Only patch if we haven't already
        if not hasattr(model_clone.model.diffusion_model, '_activation_editor_patched'):
            self._patch_transformer(model_clone.model.diffusion_model)
            model_clone.model.diffusion_model._activation_editor_patched = True

        return (model_clone,)

    def _patch_transformer(self, transformer):
        # Patch all blocks with our injection logic
        for i, block in enumerate(transformer.blocks):
            # Store original forward method
            original_forward = block.forward
            
            # Create patched forward method
            def create_patched_forward(block_idx, original_forward_func):
                def patched_forward(self, x, **kwargs):
                    # Get injection parameters from transformer options
                    transformer_options = kwargs.get('transformer_options', {})
                    activation_injections = transformer_options.get('activation_injections', [])
                    
                    if not activation_injections:
                        return original_forward_func(x, **kwargs)
                    
                    # Get current step percentage
                    current_step = kwargs.get('current_step', 0)
                    total_steps = kwargs.get('total_steps', 50)
                    current_percent = current_step / total_steps if total_steps > 0 else 0
                    
                    # Store original context for restoration
                    original_context = kwargs.get('context', None)
                    original_clip_embed = kwargs.get('clip_embed', None)
                    
                    # Check each injection
                    applied_injection = False
                    for injection in activation_injections:
                        # Check if this injection is active for this block and timestep
                        if (block_idx < len(injection['block_activations']) and 
                            injection['block_activations'][block_idx] == 1 and
                            injection['start_percent'] <= current_percent <= injection['end_percent']):
                            
                            injection_cond = injection['conditioning']
                            injection_strength = injection['strength']
                            
                            # Apply injection to cross-attention
                            if original_context is not None and injection_cond is not None:
                                # Blend original and injection conditioning
                                injected_context = injection_cond['prompt_embeds']
                                if injected_context.shape[0] != original_context.shape[0]:
                                    injected_context = injected_context.expand(original_context.shape[0], -1, -1)
                                
                                # If already injected, compound the effect
                                if applied_injection:
                                    kwargs['context'] = kwargs['context'] * (1.0 - injection_strength) + injected_context * injection_strength
                                else:
                                    kwargs['context'] = original_context * (1.0 - injection_strength) + injected_context * injection_strength
                                
                                # Also handle clip embeddings if present
                                if 'clip_embed' in injection_cond and original_clip_embed is not None:
                                    injected_clip = injection_cond['clip_embed']
                                    if injected_clip.shape[0] != original_clip_embed.shape[0]:
                                        injected_clip = injected_clip.expand(original_clip_embed.shape[0], -1, -1)
                                    
                                    if applied_injection:
                                        kwargs['clip_embed'] = kwargs['clip_embed'] * (1.0 - injection_strength) + injected_clip * injection_strength
                                    else:
                                        kwargs['clip_embed'] = original_clip_embed * (1.0 - injection_strength) + injected_clip * injection_strength
                                
                                applied_injection = True
                    
                    # Call original forward
                    result = original_forward_func(x, **kwargs)
                    
                    # Restore original context
                    if applied_injection:
                        if original_context is not None:
                            kwargs['context'] = original_context
                        if original_clip_embed is not None:
                            kwargs['clip_embed'] = original_clip_embed
                            
                    return result
                
                return patched_forward
            
            # Apply the patch
            block.forward = MethodType(create_patched_forward(i, original_forward), block)


class WanVideoBlockActivationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = { 
            "required": { 
                "preset": (["custom", "all", "none", "first_half", "second_half", "early_blocks", "mid_blocks", "late_blocks", "alternating", "sparse"],), 
            } 
        }
        optional_inputs = {}
        for i in range(40):
            optional_inputs[f"block_{i}"] = ("BOOLEAN", {"default": False})
        inputs["optional"] = optional_inputs
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("block_activations",)
    FUNCTION = "build_activations"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "Helper to build block activation patterns for WanVideoActivationEditor"

    def build_activations(self, preset, **kwargs):
        num_blocks = 40
        activations = [0] * num_blocks
        if preset == "all":
            activations = [1] * num_blocks
        elif preset == "none":
            activations = [0] * num_blocks
        elif preset == "first_half":
            activations = [1] * (num_blocks // 2) + [0] * (num_blocks // 2)
        elif preset == "second_half":
            activations = [0] * (num_blocks // 2) + [1] * (num_blocks // 2)
        elif preset == "early_blocks":
            for i in range(num_blocks // 4):
                activations[i] = 1
        elif preset == "mid_blocks":
            start = num_blocks // 4
            end = start + (num_blocks // 2)
            for i in range(start, end):
                activations[i] = 1
        elif preset == "late_blocks":
            start = num_blocks - (num_blocks // 4)
            for i in range(start, num_blocks):
                activations[i] = 1
        elif preset == "alternating":
            for i in range(num_blocks):
                if i % 2 == 0:
                    activations[i] = 1
        elif preset == "sparse":
            # Activate every 4th block
            for i in range(0, num_blocks, 4):
                activations[i] = 1
        elif preset == "custom":
            for i in range(num_blocks):
                if kwargs.get(f"block_{i}", False):
                    activations[i] = 1
        activation_string = "".join(map(str, activations))
        return (activation_string,)


class WanVideoActivationScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "schedule_mode": (["sequential", "alternating", "gradient"], {"default": "sequential"}),
            },
            "optional": {
                "injection_1": ("WANVIDEOTEXTEMBEDS",),
                "injection_1_blocks": ("STRING", {"default": "0000111100000000000000000000000000000000"}),
                "injection_1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "injection_1_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "injection_1_end": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "injection_2": ("WANVIDEOTEXTEMBEDS",),
                "injection_2_blocks": ("STRING", {"default": "0000000011110000000000000000000000000000"}),
                "injection_2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "injection_2_start": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01}),
                "injection_2_end": ("FLOAT", {"default": 0.66, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "injection_3": ("WANVIDEOTEXTEMBEDS",),
                "injection_3_blocks": ("STRING", {"default": "0000000000001111000000000000000000000000"}),
                "injection_3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "injection_3_start": ("FLOAT", {"default": 0.66, "min": 0.0, "max": 1.0, "step": 0.01}),
                "injection_3_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "schedule_injections"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "Schedule multiple condition injections at different timesteps and blocks. Enables prompt travel and complex activation experiments."

    def schedule_injections(self, model, schedule_mode, **kwargs):
        model_clone = model.clone()
        
        # Initialize transformer options
        if 'transformer_options' not in model_clone.model_options:
            model_clone.model_options['transformer_options'] = {}
        
        if 'activation_injections' not in model_clone.model_options['transformer_options']:
            model_clone.model_options['transformer_options']['activation_injections'] = []
        
        # Process each injection
        for i in range(1, 4):
            injection_cond = kwargs.get(f'injection_{i}')
            if injection_cond is not None:
                blocks_str = kwargs.get(f'injection_{i}_blocks', "0" * 40)
                try:
                    blocks = [int(bit) for bit in blocks_str]
                except ValueError:
                    raise ValueError(f"injection_{i}_blocks must be a string of 0s and 1s")
                
                strength = kwargs.get(f'injection_{i}_strength', 1.0)
                start = kwargs.get(f'injection_{i}_start', 0.0)
                end = kwargs.get(f'injection_{i}_end', 1.0)
                
                # Apply schedule mode modifications
                if schedule_mode == "alternating":
                    # Alternate strength between timesteps
                    strength = strength * (0.5 + 0.5 * (i % 2))
                elif schedule_mode == "gradient":
                    # Gradually decrease strength over time
                    strength = strength * (1.0 - (i - 1) * 0.3)
                
                model_clone.model_options['transformer_options']['activation_injections'].append({
                    'conditioning': injection_cond,
                    'strength': strength,
                    'block_activations': blocks,
                    'start_percent': start,
                    'end_percent': end
                })
        
        # Patch transformer if needed
        if not hasattr(model_clone.model.diffusion_model, '_activation_editor_patched'):
            activator = WanVideoActivationEditor()
            activator._patch_transformer(model_clone.model.diffusion_model)
            model_clone.model.diffusion_model._activation_editor_patched = True
        
        return (model_clone,)
