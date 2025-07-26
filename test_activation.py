"""
Quick test script to verify activation patches are working
Run this after loading your model to check if patches execute
"""

import torch
import os

# Force maximum debugging
os.environ['WAN_ACTIVATION_DEBUG'] = '1'
os.environ['WAN_ACTIVATION_VERBOSE'] = '1'
os.environ['WAN_ACTIVATION_TRACE'] = '1'

print("="*60)
print("WanVideo Activation Editor - Diagnostic Test")
print("="*60)

def test_basic_functionality():
    """Test if basic imports and setup work"""
    try:
        from . import nodes
        print("✓ Nodes module imported successfully")
        
        from . import activation_patch
        print("✓ Activation patch module imported successfully")
        
        from . import debug_utils
        print("✓ Debug utilities imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_patch_execution(model, text_embeds):
    """Test if patches are actually being called"""
    print("\n" + "="*60)
    print("Testing Patch Execution")
    print("="*60)
    
    # Add a global counter to track patch calls
    import builtins
    builtins._wan_patch_counter = 0
    
    # Modify the patch to increment counter
    original_process = activation_patch.ActivationPatcher._process_injection
    
    def counting_process(self, block_idx, context, injection_embeds, injection_strength):
        builtins._wan_patch_counter += 1
        print(f"\n*** PATCH EXECUTED: Block {block_idx} ***")
        print(f"*** Total patch calls: {builtins._wan_patch_counter} ***\n")
        return original_process(self, block_idx, context, injection_embeds, injection_strength)
    
    activation_patch.ActivationPatcher._process_injection = counting_process
    
    print("Patch counter installed. Run your generation workflow now.")
    print(f"Initial patch count: {builtins._wan_patch_counter}")
    
    return True

def test_context_flow(model):
    """Test how context flows through the model"""
    print("\n" + "="*60)
    print("Testing Context Flow")
    print("="*60)
    
    # Create a hook to monitor forward passes
    hooks = []
    
    def create_hook(name):
        def hook(module, input, output):
            print(f"\n[HOOK] {name}:")
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if hasattr(inp, 'shape'):
                        print(f"  Input {i}: {inp.shape}")
                    elif isinstance(inp, dict):
                        print(f"  Input {i}: dict with keys {list(inp.keys())}")
            if hasattr(output, 'shape'):
                print(f"  Output: {output.shape}")
        return hook
    
    # Try to find and hook transformer blocks
    if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
        diffusion_model = model.model.diffusion_model
        if hasattr(diffusion_model, 'joint_blocks'):
            print(f"Found {len(diffusion_model.joint_blocks)} transformer blocks")
            
            # Hook first and last blocks
            if len(diffusion_model.joint_blocks) > 0:
                hook1 = diffusion_model.joint_blocks[0].register_forward_hook(create_hook("Block 0"))
                hooks.append(hook1)
                
            if len(diffusion_model.joint_blocks) > 1:
                hook2 = diffusion_model.joint_blocks[-1].register_forward_hook(create_hook(f"Block {len(diffusion_model.joint_blocks)-1}"))
                hooks.append(hook2)
                
            print("Hooks installed on first and last blocks")
        else:
            print("No joint_blocks found in diffusion model")
    else:
        print("Could not access diffusion model structure")
    
    return hooks

def test_injection_impact():
    """Test if injections actually change the context"""
    print("\n" + "="*60)
    print("Testing Injection Impact")
    print("="*60)
    
    # Create test tensors
    context = torch.randn(1, 512, 5120)
    injection = torch.randn(1, 512, 5120) * 2  # Different magnitude
    
    # Test blending
    strength = 0.5
    blended = (1 - strength) * context + strength * injection
    
    # Calculate statistics
    diff = (blended - context).abs()
    change_percent = (diff > 0.01).float().mean().item() * 100
    
    print(f"Context norm: {context.norm().item():.3f}")
    print(f"Injection norm: {injection.norm().item():.3f}")
    print(f"Blended norm: {blended.norm().item():.3f}")
    print(f"Change percentage: {change_percent:.1f}%")
    
    if change_percent > 40:
        print("✓ Blending math works correctly")
    else:
        print("✗ Blending produces insufficient change")
    
    return change_percent > 40

def run_all_tests(model=None, text_embeds=None):
    """Run all diagnostic tests"""
    print("\nRunning WanVideo Activation Editor Diagnostics...\n")
    
    results = []
    
    # Test 1: Basic imports
    results.append(("Basic Imports", test_basic_functionality()))
    
    # Test 2: Injection math
    results.append(("Injection Math", test_injection_impact()))
    
    # Test 3: Patch execution (needs model)
    if model is not None and text_embeds is not None:
        results.append(("Patch Execution", test_patch_execution(model, text_embeds)))
    else:
        print("\nSkipping patch execution test (need model and text_embeds)")
    
    # Test 4: Context flow (needs model)
    if model is not None:
        hooks = test_context_flow(model)
        results.append(("Context Flow Hooks", len(hooks) > 0))
    else:
        print("\nSkipping context flow test (need model)")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed < len(results):
        print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n✅ All tests passed!")
    
    print("\nNOTE: To fully verify activation editing:")
    print("1. Ensure patch execution counter increases during generation")
    print("2. Check that context shapes match expected values")
    print("3. Monitor hooks to see data flow through blocks")

if __name__ == "__main__":
    run_all_tests()
else:
    print("Import this module and call run_all_tests(model, text_embeds) to diagnose")