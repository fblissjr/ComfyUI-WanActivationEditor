#!/usr/bin/env python3
"""Quick test to verify patching is working correctly."""

import os
import sys

# Force debug mode
os.environ['WAN_ACTIVATION_DEBUG'] = '1'
os.environ['WAN_ACTIVATION_VERBOSE'] = '1'

from activation_patch import get_patcher
from debug_utils import debug_print

class MockBlock:
    """Mock transformer block for testing."""
    def __init__(self, idx):
        self.idx = idx
        self._forward_called = False

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, current_step, **kwargs):
        self._forward_called = True
        return x + context * 0.1  # Simple transformation

class MockModel:
    """Mock model with blocks."""
    def __init__(self, num_blocks=40):
        self.blocks = [MockBlock(i) for i in range(num_blocks)]
        self.text_embedding = None
        self.text_len = 512

class MockModelPatcher:
    """Mock ComfyUI ModelPatcher."""
    def __init__(self):
        self.model = type('obj', (object,), {
            'diffusion_model': MockModel()
        })()
        self.model_options = {'transformer_options': {}}

    def clone(self):
        return self

def test_patching():
    """Test the patching mechanism."""
    print("\n=== Testing WanVideo Activation Patching ===\n")

    # Create mock objects
    patcher = get_patcher()
    model_patcher = MockModelPatcher()

    # Configuration
    activation_config = {
        'active_blocks': [10, 11, 12, 13, 14],  # Patch blocks 10-14
        'injection_strength': 0.5,
        'injection_embeds': None  # Would normally be embeddings
    }

    # Apply patches
    print("1. Applying patches...")
    try:
        patcher.patch_model(model_patcher, activation_config)
        print("✓ Patching completed without errors")
    except Exception as e:
        print(f"✗ Patching failed: {e}")
        return

    # Test if patches are applied
    print("\n2. Testing if patches are applied...")
    model = model_patcher.model.diffusion_model
    for idx in activation_config['active_blocks']:
        block = model.blocks[idx]
        if hasattr(block, 'forward'):
            print(f"✓ Block {idx} has forward method")
        else:
            print(f"✗ Block {idx} missing forward method")

    # Simulate forward pass
    print("\n3. Simulating forward pass...")
    import torch
    x = torch.randn(1, 100, 768)
    e = torch.randn(1, 6, 768)
    seq_lens = torch.tensor([100])
    grid_sizes = torch.tensor([[5, 20, 20]])
    freqs = torch.randn(1024, 384)
    context = torch.randn(1, 77, 768)
    current_step = 0

    for idx in activation_config['active_blocks']:
        block = model.blocks[idx]
        try:
            output = block.forward(x, e, seq_lens, grid_sizes, freqs, context, current_step)
            print(f"✓ Block {idx} forward executed successfully")
        except Exception as e:
            print(f"✗ Block {idx} forward failed: {e}")

    # Unpatch
    print("\n4. Unpatching model...")
    patcher.unpatch_model(model_patcher)
    print("✓ Model unpatched")

    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    test_patching()
