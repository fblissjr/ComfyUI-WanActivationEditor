import torch
import json
from copy import deepcopy
import gc
import weakref
from typing import Dict, Any, Optional

# Immediate debug output
print("[WanActivationEditor] Loading nodes.py module")

# We work with the model structure as passed by WanVideoWrapper
from .activation_patch import get_patcher, ContextPreprocessor

# Import debug utilities and force a test print
from .debug_utils import debug_print, DEBUG, VERBOSE
print(f"[WanActivationEditor] Debug flags: DEBUG={DEBUG}, VERBOSE={VERBOSE}")
debug_print("Debug utilities loaded and working!")

class WanVideoActivationEditor:
    """
    Main node for applying block-level activation editing to WanVideo models.
    Injects alternative text conditioning into specific transformer blocks.
    """
    
    def __init__(self):
        self.patcher = get_patcher()
        # Track patched models for cleanup
        self._patched_models = weakref.WeakSet()
    
    @classmethod
    def INPUT_TYPES(cls):
        default_activations = "0" * 40
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "injection_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "label": "Injection Strength"}),
                "block_activations": ("STRING", {"default": default_activations, "multiline": False, "label": "Block Activations"}),
                "enable_patching": ("BOOLEAN", {"default": True, "label": "Enable Runtime Patching"}),
                "log_level": (["off", "basic", "verbose", "trace"], {"default": "basic", "label": "Log Level"}),
            },
            "optional": {
                "injection_embeds": ("WANVIDEOTEXTEMBEDS",),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL", "WANVIDEOTEXTEMBEDS")
    RETURN_NAMES = ("model", "text_embeds")
    FUNCTION = "apply_activation_editing"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "Inject alternative conditioning into specific transformer blocks with runtime patching."
    
    def __del__(self):
        """Cleanup when node is deleted."""
        try:
            # Unpatch all models
            for model in list(self._patched_models):
                try:
                    self.patcher.unpatch_model(model)
                except:
                    pass
            self._patched_models.clear()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def apply_activation_editing(self, model, text_embeds, injection_strength, block_activations, 
                               injection_embeds=None, enable_patching=True, log_level="basic"):
        # Import debug utilities
        from .debug_utils import debug_print, verbose_print, debug_memory, set_log_level
        
        # Set the log level
        set_log_level(log_level)
        
        # Always print basic info to console for debugging
        if log_level != "off":
            print("\n[WanVideoActivationEditor] Processing...")
            print(f"  Injection strength: {injection_strength}")
            print(f"  Enable patching: {enable_patching}")
            print(f"  Has injection embeds: {injection_embeds is not None}")
        
        debug_print("=== WanVideoActivationEditor.apply_activation_editing called ===")
        debug_print(f"Injection strength: {injection_strength}")
        debug_print(f"Enable patching: {enable_patching}")
        debug_print(f"Block activations: {block_activations[:10]}... (first 10)")
        debug_memory("Initial ")
        
        # If no injection embeds provided, just pass through
        if injection_embeds is None:
            debug_print("No injection embeds provided, passing through")
            return (model, text_embeds)
        
        # Parse block activations
        try:
            activations = [int(bit) for bit in block_activations]
            debug_print(f"Parsed {len(activations)} block activations")
        except ValueError:
            raise ValueError("`block_activations` must be a string of 0s and 1s (e.g., '00011100000...')")
        
        # Check if any blocks are activated
        active_blocks = [i for i, a in enumerate(activations) if a == 1]
        debug_print(f"Active blocks: {active_blocks}")
        
        if log_level != "off":
            print(f"  Active blocks: {active_blocks}")
            print(f"  Total active: {len(active_blocks)}/40")
        
        if not active_blocks:
            debug_print("WARNING: No blocks are activated! Nothing will be injected.")
            debug_print("Make sure to select some blocks in WanVideoBlockActivationBuilder or pass a proper activation pattern.")
            if log_level != "off":
                print("  WARNING: No blocks activated - nothing will be injected!")
            return (model, text_embeds)
        
        # Clone the model patcher to avoid modifying the original
        m = model.clone()
        
        # Clean VRAM before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare injection context
        injection_context = None
        patching_successful = False
        
        if enable_patching and injection_embeds is not None:
            injection_context = ContextPreprocessor.prepare_injection_context(injection_embeds)
            
            # Pre-project embeddings if needed
            if injection_context is not None and torch.is_tensor(injection_context):
                # Try to pre-project embeddings through the model's text_embedding layer
                try:
                    actual_model = m.model
                    if hasattr(actual_model, 'model'):
                        actual_model = actual_model.model
                    if hasattr(actual_model, 'diffusion_model'):
                        actual_model = actual_model.diffusion_model
                    
                    text_embedding_layer = None
                    text_len = 512  # Default
                    
                    # Look for text_embedding layer
                    if hasattr(actual_model, 'text_embedding'):
                        text_embedding_layer = actual_model.text_embedding
                        if hasattr(actual_model, 'text_len'):
                            text_len = actual_model.text_len
                    elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'text_embedding'):
                        text_embedding_layer = actual_model.transformer.text_embedding
                        if hasattr(actual_model.transformer, 'text_len'):
                            text_len = actual_model.transformer.text_len
                    
                    if text_embedding_layer is not None and injection_context.shape[-1] == 4096:
                        # Pre-project the injection embeddings following WanVideoWrapper's exact method
                        with torch.no_grad():
                            orig_device = injection_context.device
                            orig_dtype = injection_context.dtype
                            
                            # Get the device and dtype from the text_embedding layer
                            text_emb_params = next(text_embedding_layer.parameters())
                            target_device = text_emb_params.device
                            target_dtype = text_emb_params.dtype
                            
                            # Ensure injection_context is 2D [seq_len, 4096]
                            if injection_context.dim() == 3 and injection_context.shape[0] == 1:
                                injection_context = injection_context.squeeze(0)
                            
                            # Convert to list format as WanVideoWrapper expects
                            injection_list = [injection_context]
                            
                            # Project through text_embedding layer using WanVideoWrapper's exact method:
                            # torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
                            padded_injection = torch.stack([
                                torch.cat([
                                    u.to(device=target_device, dtype=target_dtype), 
                                    u.new_zeros(text_len - u.size(0), u.size(1), device=target_device, dtype=target_dtype)
                                ])
                                for u in injection_list
                            ])
                            
                            # Project through text_embedding layer
                            injection_context = text_embedding_layer(padded_injection)
                            
                            # Update the injection_embeds dict with projected embeddings
                            if isinstance(injection_embeds, dict):
                                injection_embeds['prompt_embeds'] = injection_context
                            
                except Exception as e:
                    # Silently fall back to runtime projection
                    pass
            
            # Apply runtime patching
            activation_config = {
                'active_blocks': active_blocks,
                'injection_strength': injection_strength,
                'injection_embeds': injection_context if injection_context is not None and injection_context.shape[-1] == 5120 else injection_embeds
            }
            
            debug_print(f"\n=== Attempting to patch model ===")
            debug_print(f"Active blocks to patch: {active_blocks}")
            debug_print(f"Injection strength: {injection_strength}")
            
            try:
                patching_successful = self.patcher.patch_model(m, activation_config)
                debug_print(f"Patching result: {patching_successful}")
                
                if patching_successful:
                    debug_print("SUCCESS: Model patched successfully!")
                    if log_level != "off":
                        print(f"  ✓ Model patched successfully - {len(active_blocks)} blocks will be modified during generation")
                else:
                    debug_print("WARNING: Patching returned False!")
                    if log_level != "off":
                        print(f"  ⚠️ Patching may have failed - check debug output")
                
                # Track for cleanup
                self._patched_models.add(m)
                
                # Set up cleanup callback for when model is deleted
                original_del = m.__class__.__del__ if hasattr(m.__class__, '__del__') else None
                
                def cleanup_on_delete(self_model):
                    try:
                        get_patcher().unpatch_model(self_model)
                        if self_model in self._patched_models:
                            self._patched_models.discard(self_model)
                        # Clean VRAM
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    finally:
                        if original_del:
                            original_del(self_model)
                
                # Bind cleanup to model instance
                import types
                m.__del__ = types.MethodType(cleanup_on_delete, m)
                
            except Exception as e:
                patching_successful = False
                debug_print(f"ERROR during patching: {e}")
                if log_level != "off":
                    print(f"  ❌ ERROR during patching: {e}")
                import traceback
                traceback.print_exc()
        
        # Create a modified text_embeds that includes our activation information
        modified_embeds = text_embeds.copy()
        
        # Add our activation configuration to the text embeddings dict
        modified_embeds['wan_activation_editor'] = {
            'injection_embeds': injection_embeds.get('prompt_embeds') if injection_embeds else None,
            'active_blocks': active_blocks,
            'injection_strength': injection_strength,
            'block_activations': block_activations,
            'patching_enabled': enable_patching
        }
        
        # Store in transformer_options as well for the model to access
        if 'transformer_options' not in m.model_options:
            m.model_options['transformer_options'] = {}
        
        m.model_options['transformer_options']['wan_activation_editor'] = {
            'active_blocks': active_blocks,
            'injection_strength': injection_strength,
            'enabled': True,
            'runtime_patched': enable_patching and patching_successful
        }
        
        # Clean up temporary tensors
        if injection_context is not None and injection_context is not injection_embeds:
            del injection_context
        gc.collect()
        
        return (m, modified_embeds)


class WanVideoBlockActivationBuilder:
    """
    Helper node to build block activation patterns for WanVideoActivationEditor.
    Provides preset patterns and custom block selection.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = { 
            "required": { 
                "preset": ([
                    "custom", "all", "none", "first_half", "second_half", 
                    "early_blocks", "mid_blocks", "late_blocks", 
                    "alternating", "sparse"
                ],), 
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


class WanVideoBlockActivationViewer:
    """
    Debug node to view current activation state and visualize patterns.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "show_visual": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "block_activations": ("STRING",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("visualization",)
    FUNCTION = "view_activations"
    CATEGORY = "WanVideoWrapper/Experimental"
    DESCRIPTION = "View and visualize activation patterns. Connect block_activations from WanVideoBlockActivationBuilder to see the pattern."

    def view_activations(self, model, show_visual, block_activations="", text_embeds=None):
        lines = []
        lines.append("=== WanVideo Activation State ===")
        
        # Always print to console for debugging
        print("\n=== WanVideo Activation State ===")
        
        # Check model for activation info
        activation_info = None
        if 'transformer_options' in model.model_options:
            if 'wan_activation_editor' in model.model_options['transformer_options']:
                activation_info = model.model_options['transformer_options']['wan_activation_editor']
        
        if activation_info:
            lines.append("\n[Model Configuration]")
            lines.append(f"Active blocks: {activation_info.get('active_blocks', [])}")
            lines.append(f"Injection strength: {activation_info.get('injection_strength', 0)}")
            lines.append(f"Enabled: {activation_info.get('enabled', False)}")
            lines.append(f"Runtime patched: {activation_info.get('runtime_patched', False)}")
            
            # Always print to console
            print(f"Active blocks: {activation_info.get('active_blocks', [])}")
            print(f"Injection strength: {activation_info.get('injection_strength', 0)}")
            print(f"Runtime patched: {activation_info.get('runtime_patched', False)}")
            
            if activation_info.get('runtime_patched'):
                lines.append("\n[Runtime Patching Status]")
                lines.append("✓ Transformer blocks are patched")
                lines.append("✓ Injection will be applied during generation")
                print("✓ Transformer blocks are patched")
            else:
                lines.append("\n[Runtime Patching Status]")
                lines.append("✗ Patching disabled - injection configured but not active")
                print("✗ Patching disabled - injection configured but not active")
        else:
            lines.append("\n[Model Configuration]")
            lines.append("No activation editor modifications found")
            print("No activation editor modifications found")
        
        # Visualize block pattern if provided
        if not block_activations:
            lines.append("\n[Tip: Connect block_activations from WanVideoBlockActivationBuilder]")
            lines.append("[to visualize the activation pattern]")
        elif show_visual:
            lines.append("\n[Block Pattern Visualization]")
            try:
                activations = [int(bit) for bit in block_activations]
                
                # Visual representation
                for group in range(4):
                    start = group * 10
                    end = min(start + 10, len(activations))
                    
                    # Block indices
                    indices = "".join(f"{i:3}" for i in range(start, end))
                    lines.append(indices)
                    
                    # Visual blocks
                    visual = ""
                    for i in range(start, end):
                        visual += " ■ " if activations[i] else " □ "
                    lines.append(visual)
                    lines.append("")
                
                # Summary
                active_blocks = [i for i, a in enumerate(activations) if a == 1]
                lines.append(f"Active blocks: {active_blocks}")
                lines.append(f"Total active: {len(active_blocks)}/{len(activations)}")
                
                # Always print active blocks to console
                print(f"Active blocks from pattern: {active_blocks}")
                print(f"Total active: {len(active_blocks)}/{len(activations)}")
                
                # Pattern detection
                if len(active_blocks) > 0:
                    lines.append("\n[Pattern Analysis]")
                    if active_blocks == list(range(10)):
                        lines.append("Pattern: Early blocks (0-9)")
                    elif active_blocks == list(range(30, 40)):
                        lines.append("Pattern: Late blocks (30-39)")
                    elif active_blocks == list(range(10, 30)):
                        lines.append("Pattern: Mid blocks (10-29)")
                    elif all(b % 2 == 0 for b in active_blocks):
                        lines.append("Pattern: Even blocks")
                    elif all(b % 2 == 1 for b in active_blocks):
                        lines.append("Pattern: Odd blocks")
                    else:
                        lines.append("Pattern: Custom")
                
            except ValueError:
                lines.append("Invalid block activation string")
        
        # Add debug information if text_embeds provided
        if text_embeds is not None and 'wan_activation_editor' in text_embeds:
            lines.append("\n[Embedding Information]")
            
            inj_config = text_embeds['wan_activation_editor']
            lines.append(f"Injection embeds present: {inj_config.get('injection_embeds') is not None}")
            lines.append(f"Configured strength: {inj_config.get('injection_strength', 0)}")
            
            # If verbose logging is enabled globally, show more details
            from .debug_utils import VERBOSE
            if VERBOSE and inj_config.get('injection_embeds') is not None:
                inj_embeds = inj_config['injection_embeds']
                if hasattr(inj_embeds, 'shape'):
                    lines.append(f"Injection shape: {inj_embeds.shape}")
                    lines.append(f"Injection device: {inj_embeds.device}")
                    lines.append(f"Injection dtype: {inj_embeds.dtype}")
        
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {
    "WanVideoActivationEditor": WanVideoActivationEditor,
    "WanVideoBlockActivationBuilder": WanVideoBlockActivationBuilder,
    "WanVideoBlockActivationViewer": WanVideoBlockActivationViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoActivationEditor": "WanVideo Activation Editor",
    "WanVideoBlockActivationBuilder": "WanVideo Block Activation Builder",
    "WanVideoBlockActivationViewer": "WanVideo Block Activation Viewer",
}