"""
Alternative injection nodes that work without modifying WanVideoWrapper.
These use different strategies to achieve activation-like effects, but likely don't work as well as current implementation.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import gc

from .database import get_db


class WanVideoLatentInjector:
    """
    Inject activations by manipulating latents during the denoising process.
    This works by creating two denoising paths and blending them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "main_latent": ("LATENT",),
                "injection_latent": ("LATENT",),
                "injection_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 100}),
                "end_at_step": ("INT", {"default": 20, "min": 0, "max": 100}),
                "blend_mode": (["linear", "sigmoid", "cosine"], {"default": "linear"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("blended_latent",)
    FUNCTION = "inject_latents"
    CATEGORY = "WanVideoWrapper/Alternative"
    DESCRIPTION = "Blend two latent paths to achieve activation-like effects"

    def inject_latents(self, model, main_latent, injection_latent, injection_strength,
                      start_at_step, end_at_step, blend_mode):
        """Blend latents to create hybrid denoising trajectory."""
        print("\n[LatentInjector] === Starting Latent Injection ===")
        print(f"[LatentInjector] Blend mode: {blend_mode}")
        print(f"[LatentInjector] Injection strength: {injection_strength}")
        print(f"[LatentInjector] Steps: {start_at_step} to {end_at_step}")

        # Extract latent tensors
        main_samples = main_latent["samples"]
        injection_samples = injection_latent["samples"]

        # Ensure compatible shapes
        if main_samples.shape != injection_samples.shape:
            print(f"[LatentInjector] Shape mismatch - main: {main_samples.shape}, injection: {injection_samples.shape}")
            # Attempt to match shapes
            if injection_samples.shape[0] != main_samples.shape[0]:
                injection_samples = injection_samples.expand(main_samples.shape[0], -1, -1, -1, -1)

        # Calculate blend weights based on mode
        if blend_mode == "linear":
            weight = injection_strength
        elif blend_mode == "sigmoid":
            # Smooth transition
            x = np.linspace(-6, 6, end_at_step - start_at_step)
            weights = 1 / (1 + np.exp(-x))
            weight = weights.mean() * injection_strength
        elif blend_mode == "cosine":
            # Cosine interpolation
            weight = injection_strength * 0.5 * (1 - np.cos(np.pi * 0.5))

        # Blend latents
        with torch.no_grad():
            blended_samples = (1 - weight) * main_samples + weight * injection_samples

        # Create output
        blended_latent = {
            "samples": blended_samples,
            "injection_info": {
                "strength": injection_strength,
                "start_step": start_at_step,
                "end_step": end_at_step,
                "blend_mode": blend_mode
            }
        }

        print(f"[LatentInjector] Created blended latent with weight {weight:.3f}")
        print("[LatentInjector] === Latent Injection Complete ===\n")

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (blended_latent,)


class WanVideoGuidanceController:
    """
    Control the generation by manipulating classifier-free guidance.
    This creates custom guidance that blends different conditioning paths.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "main_conditioning": ("WANVIDEOTEXTEMBEDS",),
                "injection_conditioning": ("WANVIDEOTEXTEMBEDS",),
                "cfg_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "injection_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "blend_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("GUIDANCE_CONFIG",)
    RETURN_NAMES = ("guidance_config",)
    FUNCTION = "configure_guidance"
    CATEGORY = "WanVideoWrapper/Alternative"
    DESCRIPTION = "Create mixed guidance for pseudo-activation effects"

    def configure_guidance(self, model, main_conditioning, injection_conditioning,
                         cfg_scale, injection_scale, blend_start, blend_end):
        """Configure custom guidance blending."""
        print("\n[GuidanceController] === Configuring Guidance ===")
        print(f"[GuidanceController] Main CFG scale: {cfg_scale}")
        print(f"[GuidanceController] Injection scale: {injection_scale}")
        print(f"[GuidanceController] Blend range: {blend_start:.2f} to {blend_end:.2f}")

        # Create guidance configuration
        guidance_config = {
            "type": "mixed_guidance",
            "main_conditioning": main_conditioning,
            "injection_conditioning": injection_conditioning,
            "cfg_scale": cfg_scale,
            "injection_scale": injection_scale,
            "blend_start": blend_start,
            "blend_end": blend_end,
            "active": True
        }

        # Store in model options for sampler to use
        m = model.clone()
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}

        m.model_options['transformer_options']['guidance_controller'] = guidance_config

        print("[GuidanceController] Guidance configuration stored")
        print("[GuidanceController] === Configuration Complete ===\n")

        return (guidance_config,)


class WanVideoNoiseController:
    """
    Control generation by using structured noise patterns.
    Different noise in different regions creates pseudo-activation effects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "frames": ("INT", {"default": 81, "min": 1, "max": 200}),
                "main_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "injection_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "noise_pattern": (["uniform", "spatial_blocks", "temporal_blocks", "checkerboard", "radial"],
                                {"default": "spatial_blocks"}),
                "pattern_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("structured_noise",)
    FUNCTION = "create_structured_noise"
    CATEGORY = "WanVideoWrapper/Alternative"
    DESCRIPTION = "Create structured noise patterns for pseudo-activation"

    def create_structured_noise(self, width, height, frames, main_seed, injection_seed,
                              noise_pattern, pattern_strength):
        """Create noise with different patterns in different regions."""
        print("\n[NoiseController] === Creating Structured Noise ===")
        print(f"[NoiseController] Pattern: {noise_pattern}")
        print(f"[NoiseController] Pattern strength: {pattern_strength}")

        # Calculate latent dimensions
        lat_h = height // 8
        lat_w = width // 8
        lat_frames = (frames - 1) // 4 + 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate base noises
        main_gen = torch.Generator(device="cpu").manual_seed(main_seed)
        inject_gen = torch.Generator(device="cpu").manual_seed(injection_seed)

        shape = (1, 16, lat_frames, lat_h, lat_w)
        main_noise = torch.randn(shape, generator=main_gen, device="cpu")
        inject_noise = torch.randn(shape, generator=inject_gen, device="cpu")

        # Create pattern mask
        mask = torch.zeros(shape, device="cpu")

        if noise_pattern == "uniform":
            mask.fill_(pattern_strength)

        elif noise_pattern == "spatial_blocks":
            # Divide spatially into blocks
            block_h = lat_h // 4
            block_w = lat_w // 4
            for i in range(0, lat_h, block_h * 2):
                for j in range(0, lat_w, block_w * 2):
                    mask[:, :, :, i:i+block_h, j:j+block_w] = pattern_strength

        elif noise_pattern == "temporal_blocks":
            # Alternate temporal blocks
            for t in range(0, lat_frames, 2):
                mask[:, :, t, :, :] = pattern_strength

        elif noise_pattern == "checkerboard":
            # Fine checkerboard pattern
            for i in range(lat_h):
                for j in range(lat_w):
                    if (i + j) % 2 == 0:
                        mask[:, :, :, i, j] = pattern_strength

        elif noise_pattern == "radial":
            # Radial gradient from center
            center_h, center_w = lat_h // 2, lat_w // 2
            for i in range(lat_h):
                for j in range(lat_w):
                    dist = np.sqrt((i - center_h)**2 + (j - center_w)**2)
                    max_dist = np.sqrt(center_h**2 + center_w**2)
                    mask[:, :, :, i, j] = pattern_strength * (1 - dist / max_dist)

        # Blend noises
        structured_noise = (1 - mask) * main_noise + mask * inject_noise

        # Move to device
        structured_noise = structured_noise.to(device)

        # Create latent dict
        latent = {
            "samples": structured_noise,
            "noise_pattern": noise_pattern,
            "pattern_strength": pattern_strength
        }

        print(f"[NoiseController] Created {noise_pattern} noise pattern")
        print("[NoiseController] === Structured Noise Complete ===\n")

        # Store in database for analysis
        db = get_db()
        db.store_embedding(
            f"Structured noise: {noise_pattern} @ {pattern_strength}",
            {"prompt_embeds": [structured_noise]}
        )

        return (latent,)


class WanVideoSequentialMixer:
    """
    Mix embeddings sequentially over time/steps.
    Creates a transition effect similar to prompt travel but for activations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding_1": ("WANVIDEOTEXTEMBEDS",),
                "embedding_2": ("WANVIDEOTEXTEMBEDS",),
                "mix_schedule": (["linear", "ease_in", "ease_out", "ease_in_out", "step"],
                               {"default": "linear"}),
                "transition_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_sharpness": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "embedding_3": ("WANVIDEOTEXTEMBEDS",),
                "transition_point_2": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("WANVIDEOTEXTEMBEDS",)
    RETURN_NAMES = ("mixed_sequence",)
    FUNCTION = "create_sequence"
    CATEGORY = "WanVideoWrapper/Alternative"
    DESCRIPTION = "Create embedding sequence that transitions over generation steps"

    def create_sequence(self, embedding_1, embedding_2, mix_schedule, transition_point,
                       transition_sharpness, embedding_3=None, transition_point_2=0.75):
        """Create a sequence of embeddings that transition over time."""
        print("\n[SequentialMixer] === Creating Embedding Sequence ===")
        print(f"[SequentialMixer] Schedule: {mix_schedule}")
        print(f"[SequentialMixer] Transition at: {transition_point}")

        # Extract tensors
        def extract_tensor(embed):
            if isinstance(embed, dict):
                t = embed.get('prompt_embeds')
                return t[0] if isinstance(t, list) else t
            return embed

        tensor_1 = extract_tensor(embedding_1)
        tensor_2 = extract_tensor(embedding_2)
        tensor_3 = extract_tensor(embedding_3) if embedding_3 else None

        # Create transition schedule
        num_steps = 20  # Typical denoising steps
        weights = np.zeros((num_steps, 3 if tensor_3 else 2))

        for i in range(num_steps):
            t = i / (num_steps - 1)

            if mix_schedule == "linear":
                if tensor_3:
                    if t < transition_point:
                        weights[i, 0] = 1 - (t / transition_point)
                        weights[i, 1] = t / transition_point
                    else:
                        rel_t = (t - transition_point) / (transition_point_2 - transition_point)
                        weights[i, 1] = 1 - rel_t
                        weights[i, 2] = rel_t
                else:
                    weights[i, 0] = 1 - t
                    weights[i, 1] = t

            elif mix_schedule == "ease_in":
                eased_t = t * t
                weights[i, 0] = 1 - eased_t
                weights[i, 1] = eased_t

            elif mix_schedule == "ease_out":
                eased_t = 1 - (1 - t) * (1 - t)
                weights[i, 0] = 1 - eased_t
                weights[i, 1] = eased_t

            elif mix_schedule == "ease_in_out":
                eased_t = 3 * t * t - 2 * t * t * t
                weights[i, 0] = 1 - eased_t
                weights[i, 1] = eased_t

            elif mix_schedule == "step":
                if t < transition_point:
                    weights[i, 0] = 1
                else:
                    weights[i, 1] = 1

        # Create mixed embedding (using mid-point as representative)
        mid_idx = num_steps // 2
        if tensor_3:
            mixed = (weights[mid_idx, 0] * tensor_1 +
                    weights[mid_idx, 1] * tensor_2 +
                    weights[mid_idx, 2] * tensor_3)
        else:
            mixed = weights[mid_idx, 0] * tensor_1 + weights[mid_idx, 1] * tensor_2

        # Create output with schedule info
        mixed_sequence = embedding_1.copy() if isinstance(embedding_1, dict) else {}
        mixed_sequence['prompt_embeds'] = [mixed]
        mixed_sequence['sequence_schedule'] = weights.tolist()
        mixed_sequence['mix_schedule'] = mix_schedule
        mixed_sequence['transition_points'] = [transition_point, transition_point_2] if tensor_3 else [transition_point]

        print(f"[SequentialMixer] Created {num_steps}-step sequence")
        print("[SequentialMixer] === Sequence Complete ===\n")

        return (mixed_sequence,)


# Node mappings
ALTERNATIVE_NODE_CLASS_MAPPINGS = {
    "WanVideoLatentInjector": WanVideoLatentInjector,
    "WanVideoGuidanceController": WanVideoGuidanceController,
    "WanVideoNoiseController": WanVideoNoiseController,
    "WanVideoSequentialMixer": WanVideoSequentialMixer,
}

ALTERNATIVE_NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoLatentInjector": "WanVideo Latent Injector",
    "WanVideoGuidanceController": "WanVideo Guidance Controller",
    "WanVideoNoiseController": "WanVideo Noise Controller",
    "WanVideoSequentialMixer": "WanVideo Sequential Mixer",
}
