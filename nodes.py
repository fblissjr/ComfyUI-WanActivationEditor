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
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "alpha": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "delta": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "block_activations": ("STRING", {"default": default_activations, "multiline": False}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    FUNCTION = "apply_activation_editing"
    CATEGORY = "AdvConditioning/WanVideo"

    def apply_activation_editing(self, model, positive_conditioning, strength, alpha, delta, block_activations):
        model_clone = model.clone()
        transformer = model_clone.model.diffusion_model

        # Store the original forward method to be called from our patch
        original_forward_func = transformer.forward

        # Prepare the injected conditioning tensor
        cond_tensor = positive_conditioning["prompt_embeds"][0].to(transformer.device, dtype=transformer.dtype)

        try:
            activations = [int(bit) for bit in block_activations]
        except ValueError:
            raise ValueError("`block_activations` must be a string of 0s and 1s.")

        # This is our new, complete forward function for the whole WanModel
        def patched_model_forward(x_in, *args, **kwargs):
            # This 'x' is the list of tensors the model expects
            x = x_in[0]

            # Replicate the setup from the original forward method
            e0 = kwargs['e']
            x_ref_attn_map = kwargs.get('x_ref_attn_map', None)
            human_num = kwargs.get('human_num', 0)

            # --- Main Block Loop ---
            for i, block in enumerate(transformer.blocks):
                kwargs['block_id'] = i

                # Check if this block should be actively modified
                if i < len(activations) and activations[i] == 1:
                    # --- Active Block Logic (Man-in-the-Middle) ---
                    e_block = block.get_mod(e0)

                    # 1. Self-Attention Manipulation
                    normed_x = block.norm1(x)
                    modulated_x = block.modulate(normed_x, e_block)

                    self_attn_module = block.self_attn

                    # Get original q, k, v
                    q = self_attn_module.norm_q(self_attn_module.q(modulated_x))
                    k = self_attn_module.norm_k(self_attn_module.k(modulated_x))
                    v = self_attn_module.v(modulated_x)

                    # Create injected k, v from our conditioning
                    k_inj = self_attn_module.norm_k(self_attn_module.k(cond_tensor))
                    v_inj = self_attn_module.v(cond_tensor)

                    # Blend k and v
                    # The length of k_inj might not match k, so we slice or pad k_inj
                    if k.shape[1] > k_inj.shape[1]: # if latent is longer than cond
                        k_inj = torch.cat([k_inj, torch.zeros_like(k[:, k_inj.shape[1]:, :])], dim=1)
                        v_inj = torch.cat([v_inj, torch.zeros_like(v[:, v_inj.shape[1]:, :])], dim=1)
                    else: # if cond is longer than latent
                        k_inj = k_inj[:, :k.shape[1], :]
                        v_inj = v_inj[:, :v.shape[1], :]

                    blended_k = k * (1.0 - alpha) + k_inj * alpha
                    blended_v = v * (1.0 - alpha) + v_inj * alpha

                    # Reshape for attention
                    q = q.view(x.shape[0], -1, self_attn_module.num_heads, self_attn_module.head_dim)
                    blended_k = blended_k.view(x.shape[0], -1, self_attn_module.num_heads, self_attn_module.head_dim)
                    blended_v = blended_v.view(x.shape[0], -1, self_attn_module.num_heads, self_attn_module.head_dim)

                    # Perform attention with modified k,v
                    attn_output, x_ref_attn_map = self_attn_module(modulated_x, **kwargs)
                    attn_output = attn_output.flatten(2)
                    y = self_attn_module.o(attn_output)

                    x = x + (y * e_block[2])

                    # 2. Proceed with the rest of the block's logic (cross-attn, ffn)
                    x = block.cross_attn_ffn(x, **kwargs, human_num=human_num, x_ref_attn_map=x_ref_attn_map)

                else:
                    # --- Inactive Block Logic (Passthrough) ---
                    if multitalk_audio is not None and human_num > 0:
                        x, x_ref_attn_map = block(x, **kwargs, x_ref_attn_map=x_ref_attn_map, human_num=human_num)
                    else:
                        x = block(x, **kwargs)

            # --- After Loop ---
            # Replicate the final part of the original forward method
            x = transformer.head(x, e0.to(x.device))
            x = transformer.unpatchify(x, kwargs['grid_sizes'])
            x = [u.float() for u in x]

            return x, kwargs.get('pred_id', None)

        # Apply the patch to the model instance
        transformer.forward = patched_model_forward

        return (model_clone,)


class WanVideoBlockActivationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = { "required": { "preset": (["custom", "all", "none", "first_half", "second_half", "early_blocks", "mid_blocks", "late_blocks"],), } }
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
