import torch
import comfy.model_management as mm

class WanVideoActivationEditor:
    @classmethod
    def INPUT_TYPES(cls):
        default_activations = "0" * 40
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "positive_conditioning": ("WANVIDEOTEXTEMBEDS",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "block_activations": ("STRING", {"default": default_activations, "multiline": False}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "apply_block_conditioning"
    CATEGORY = "AdvConditioning/WanVideo"

    def apply_block_conditioning(self, model, positive_conditioning, strength, alpha, delta, block_activations):
        model_clone = model.clone()
        transformer = model_clone.model.diffusion_model

        cond_tensor = positive_conditioning["prompt_embeds"][0].to(transformer.device, dtype=transformer.dtype)
        cond_lens = torch.tensor([cond_tensor.shape[1]], device=transformer.device)

        try:
            activations = [int(bit) for bit in block_activations]
        except ValueError:
            raise ValueError("`block_activations` must be a string containing only '0' and '1'.")

        for i, block in enumerate(transformer.blocks):
            if i < len(activations) and activations[i] == 1:
                def make_patched_forward(original_forward_func, block_instance, new_cond, new_cond_lens, str_val, alpha_val, delta_val):

                    def patched_forward(*args, **kwargs):
                        # --- THE DEFINITIVE FIX ---
                        # Call the original function and capture its result.
                        original_result = original_forward_func(*args, **kwargs)

                        # Check if the result is a tuple (meaning it returned two values)
                        if isinstance(original_result, tuple):
                            original_output, original_attn_map = original_result
                        else:
                            original_output = original_result
                            original_attn_map = None # Set to None if not returned

                        normed_output = block_instance.norm3(original_output)

                        extra_cond_output = block_instance.cross_attn(
                            normed_output,
                            context=new_cond,
                            context_lens=new_cond_lens,
                            is_uncond=kwargs.get('is_uncond', False)
                        )

                        final_output = original_output * (1.0 - alpha_val) + (extra_cond_output * (str_val / delta_val)) * alpha_val

                        # Replicate the original's conditional return signature.
                        if original_attn_map is not None:
                            return final_output, original_attn_map
                        else:
                            return final_output

                    return patched_forward

                block.forward = make_patched_forward(
                    block.forward,
                    block,
                    cond_tensor,
                    cond_lens,
                    strength,
                    alpha,
                    delta
                )
                print(f"[Info] WanVideoBlockCondition: Patched block {i}.")

        return (model_clone,)

class WanVideoBlockActivationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "preset": (["custom", "all", "none", "first_half", "second_half", "early_blocks", "mid_blocks", "late_blocks"],),
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
    CATEGORY = "AdvConditioning/WanVideo"

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
        elif preset == "custom":
            for i in range(num_blocks):
                if kwargs.get(f"block_{i}", False):
                    activations[i] = 1

        activation_string = "".join(map(str, activations))
        return (activation_string,)
