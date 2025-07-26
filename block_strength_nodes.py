"""
Advanced block strength control nodes for fine-grained activation editing.
Allows different injection strengths for different transformer blocks.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import json


class WanVideoBlockStrengthBuilder:
    """
    Build custom strength patterns for each transformer block.
    Provides presets and mathematical functions for strength gradients.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": ([
                    "custom", "uniform", "linear_decay", "linear_rise",
                    "gaussian_peak", "inverse_gaussian", "sine_wave",
                    "stepped", "random", "focus_early", "focus_mid", "focus_late"
                ],),
                "base_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # For gaussian patterns
                "peak_block": ("INT", {"default": 20, "min": 0, "max": 39}),
                "spread": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 20.0, "step": 0.5}),
                # For stepped patterns  
                "step_size": ("INT", {"default": 5, "min": 1, "max": 20}),
                # For custom per-block control
                **{f"block_{i}_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}) 
                   for i in range(40)}
            }
        }
    
    RETURN_TYPES = ("BLOCK_STRENGTHS", "STRING")
    RETURN_NAMES = ("block_strengths", "strength_visualization")
    FUNCTION = "build_strengths"
    CATEGORY = "WanVideoWrapper/Advanced"
    DESCRIPTION = "Create per-block injection strength patterns"
    
    def build_strengths(self, pattern, base_strength, peak_block=20, spread=5.0, step_size=5, **kwargs):
        """Generate strength array based on pattern."""
        
        num_blocks = 40
        strengths = np.zeros(num_blocks)
        
        if pattern == "uniform":
            strengths.fill(base_strength)
            
        elif pattern == "linear_decay":
            strengths = np.linspace(base_strength, 0, num_blocks)
            
        elif pattern == "linear_rise":
            strengths = np.linspace(0, base_strength, num_blocks)
            
        elif pattern == "gaussian_peak":
            x = np.arange(num_blocks)
            strengths = base_strength * np.exp(-((x - peak_block) ** 2) / (2 * spread ** 2))
            
        elif pattern == "inverse_gaussian":
            x = np.arange(num_blocks)
            gaussian = np.exp(-((x - peak_block) ** 2) / (2 * spread ** 2))
            strengths = base_strength * (1 - gaussian)
            
        elif pattern == "sine_wave":
            x = np.arange(num_blocks)
            strengths = base_strength * (0.5 + 0.5 * np.sin(2 * np.pi * x / num_blocks))
            
        elif pattern == "stepped":
            for i in range(0, num_blocks, step_size * 2):
                strengths[i:i + step_size] = base_strength
                
        elif pattern == "random":
            strengths = np.random.uniform(0, base_strength, num_blocks)
            
        elif pattern == "focus_early":
            strengths[:10] = base_strength
            strengths[10:] = base_strength * 0.1
            
        elif pattern == "focus_mid":
            strengths[10:30] = base_strength
            strengths[:10] = base_strength * 0.1
            strengths[30:] = base_strength * 0.1
            
        elif pattern == "focus_late":
            strengths[30:] = base_strength
            strengths[:30] = base_strength * 0.1
            
        elif pattern == "custom":
            # Use individual block strengths
            for i in range(num_blocks):
                key = f"block_{i}_strength"
                if key in kwargs:
                    strengths[i] = kwargs[key]
        
        # Create visualization
        viz_lines = ["=== Block Strength Pattern ===\n"]
        
        # ASCII bar chart
        max_width = 50
        for i in range(0, num_blocks, 2):  # Show every other block to fit
            strength = strengths[i]
            bar_width = int(strength * max_width)
            bar = "█" * bar_width + "░" * (max_width - bar_width)
            viz_lines.append(f"Block {i:2d}: {bar} {strength:.2f}")
        
        # Summary stats
        viz_lines.append(f"\nPattern: {pattern}")
        viz_lines.append(f"Average strength: {np.mean(strengths):.3f}")
        viz_lines.append(f"Active blocks (>0.1): {np.sum(strengths > 0.1)}")
        
        visualization = "\n".join(viz_lines)
        
        # Convert to list for storage
        strength_dict = {
            "strengths": strengths.tolist(),
            "pattern": pattern,
            "metadata": {
                "base_strength": base_strength,
                "peak_block": peak_block if pattern in ["gaussian_peak", "inverse_gaussian"] else None,
                "spread": spread if pattern in ["gaussian_peak", "inverse_gaussian"] else None,
            }
        }
        
        return (strength_dict, visualization)


class WanVideoAdvancedActivationEditor:
    """
    Enhanced activation editor that supports per-block strength control.
    Extends the basic activation editor with fine-grained control.
    """
    
    def __init__(self):
        from .activation_patch import get_patcher
        self.patcher = get_patcher()
        self._patched_models = weakref.WeakSet()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "block_activations": ("STRING", {"default": "0" * 40}),
                "enable_patching": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "injection_embeds": ("WANVIDEOTEXTEMBEDS",),
                "block_strengths": ("BLOCK_STRENGTHS",),
                # Fallback if no block_strengths provided
                "uniform_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "text_embeds")
    FUNCTION = "apply_advanced_editing"
    CATEGORY = "WanVideoWrapper/Advanced"
    DESCRIPTION = "Apply activation editing with per-block strength control"
    
    def apply_advanced_editing(self, model, text_embeds, block_activations, enable_patching=True,
                             injection_embeds=None, block_strengths=None, uniform_strength=0.5):
        """Apply activation editing with per-block strengths."""
        
        # If no injection embeds, pass through
        if injection_embeds is None:
            return (model, text_embeds)
        
        # Parse block activations
        try:
            activations = [int(bit) for bit in block_activations]
        except ValueError:
            raise ValueError("block_activations must be a string of 0s and 1s")
        
        # Get active blocks
        active_blocks = [i for i, a in enumerate(activations) if a == 1]
        if not active_blocks:
            return (model, text_embeds)
        
        # Extract strengths
        if block_strengths and isinstance(block_strengths, dict):
            strengths = block_strengths.get('strengths', [])
            if len(strengths) != 40:
                strengths = [uniform_strength] * 40
        else:
            strengths = [uniform_strength] * 40
        
        # This is where we'd need to modify the activation_patch.py
        # to support per-block strengths. For now, let's prepare the config:
        
        activation_config = {
            'active_blocks': active_blocks,
            'block_strengths': {i: strengths[i] for i in active_blocks},
            'injection_embeds': injection_embeds,
            # Keep backward compatibility
            'injection_strength': uniform_strength,
        }
        
        # Clone model and apply enhanced patching
        m = model.clone()
        
        # The actual patching would need updates to support per-block strengths
        # For now, this prepares the structure
        
        # Store in transformer_options
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}
        
        m.model_options['transformer_options']['wan_activation_editor_advanced'] = {
            'active_blocks': active_blocks,
            'block_strengths': {i: strengths[i] for i in active_blocks},
            'enabled': True,
        }
        
        # Modified embeddings with config
        modified_embeds = text_embeds.copy()
        modified_embeds['wan_activation_editor_advanced'] = activation_config
        
        return (m, modified_embeds)


class WanVideoStrengthVisualizer:
    """
    Visualize block strength patterns for debugging and understanding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "block_strengths": ("BLOCK_STRENGTHS",),
                "show_numbers": ("BOOLEAN", {"default": True}),
                "graph_height": ("INT", {"default": 10, "min": 5, "max": 20}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text_visualization", "graph_image")
    FUNCTION = "visualize"
    CATEGORY = "WanVideoWrapper/Advanced"
    DESCRIPTION = "Visualize block strength patterns"
    
    def visualize(self, block_strengths, show_numbers, graph_height):
        """Create text and image visualizations."""
        
        if not isinstance(block_strengths, dict) or 'strengths' not in block_strengths:
            return ("Invalid block strengths data", None)
        
        strengths = block_strengths['strengths']
        pattern = block_strengths.get('pattern', 'unknown')
        
        # Text visualization
        lines = [f"=== Block Strength Visualization ({pattern}) ===\n"]
        
        # Create ASCII graph
        for y in range(graph_height, -1, -1):
            threshold = y / graph_height
            line = f"{threshold:3.1f} |"
            
            for x in range(40):
                if strengths[x] >= threshold:
                    line += "█"
                else:
                    line += " "
            
            lines.append(line)
        
        # X-axis
        lines.append("    +" + "-" * 40)
        lines.append("     " + "".join([str(i // 10) if i % 10 == 0 else " " for i in range(40)]))
        lines.append("     " + "".join([str(i % 10) for i in range(40)]))
        
        # Stats
        lines.append(f"\nBlocks with strength > 0.5: {[i for i, s in enumerate(strengths) if s > 0.5]}")
        lines.append(f"Average strength: {np.mean(strengths):.3f}")
        lines.append(f"Peak strength: {max(strengths):.3f} at block {np.argmax(strengths)}")
        
        # Create image visualization (optional, requires PIL)
        try:
            from PIL import Image, ImageDraw
            
            # Create gradient image
            width = 400
            height = 100
            img = Image.new('RGB', (width, height), 'black')
            draw = ImageDraw.Draw(img)
            
            block_width = width // 40
            for i, strength in enumerate(strengths):
                color_value = int(255 * strength)
                color = (color_value, color_value, color_value)
                x1 = i * block_width
                x2 = (i + 1) * block_width
                draw.rectangle([x1, 0, x2, height], fill=color)
                
                # Add block numbers
                if show_numbers and i % 5 == 0:
                    draw.text((x1 + 2, height - 20), str(i), fill='red')
            
            # Convert to tensor for ComfyUI
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
        except ImportError:
            img_tensor = None
        
        return ("\n".join(lines), img_tensor)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "WanVideoBlockStrengthBuilder": WanVideoBlockStrengthBuilder,
    "WanVideoAdvancedActivationEditor": WanVideoAdvancedActivationEditor,
    "WanVideoStrengthVisualizer": WanVideoStrengthVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoBlockStrengthBuilder": "WanVideo Block Strength Builder",
    "WanVideoAdvancedActivationEditor": "WanVideo Advanced Activation Editor",
    "WanVideoStrengthVisualizer": "WanVideo Strength Visualizer",
}