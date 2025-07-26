# WanVideo Activation Editor User Guide

## Overview

Block-level prompt injection for WanVideo models. Inject alternative text conditioning into specific transformer blocks to blend different concepts at various processing depths.

- **What it does**: Mixes different prompts at specific transformer blocks
- **What it doesn't do**: Change prompts over time (not temporal prompt travel)
- **Key mechanism**: 40 transformer blocks, each can use different text embeddings

## How It Works

1. Text → T5 encoder → embeddings (4096-dim)
2. Projection → transformer space (5120-dim)
3. 40 transformer blocks process sequentially
4. Each block can use different text embeddings via cross-attention
5. Blended embeddings guide generation

The editor stores configuration in `transformer_options` for WanVideoWrapper to process during generation.

## Quick Start

1. Two `WanVideoTextEncode` nodes: main prompt + injection prompt
2. `WanVideoBlockActivationBuilder`: select preset or custom blocks
3. `WanVideoActivationEditor`: set strength (0.3-0.5) and log level
4. Continue normal generation

**Critical**: Prompts must be >50% different for visible effects!

Example:
```
Main: "A gray tabby cat walks through a sunlit garden. The cat moves gracefully between 
      rows of colorful flowers, its tail swaying gently. Butterflies flutter nearby."
      
Injection: "An underwater coral reef scene with crystal clear water. Schools of tropical 
           fish swim between vibrant coral formations. Sunlight filters through the water."
           
Blocks: late_blocks
Strength: 0.4
Result: Cat with aquatic features, garden with underwater lighting effects
```

## Block Patterns

### General Tendencies (not rigid rules)
- **0-10**: Often affects colors, textures, low-level features
- **10-20**: May influence shapes, local structures
- **20-30**: Can affect objects, composition
- **30-40**: Typically handles semantic meaning

### Presets
- `early_blocks` (0-10): Texture/color experiments
- `mid_blocks` (10-30): Structure blending
- `late_blocks` (30-40): Semantic mixing
- `alternating`: Every 2nd block
- `sparse`: Every 4th block
- `first_half` (0-20): Foundation layer
- `second_half` (20-40): Higher concepts

## Workflow Setup

1. Load model with `WanVideoModelLoader`
2. Encode main and injection prompts with `WanVideoTextEncode`
3. Select blocks with `WanVideoBlockActivationBuilder`
4. Apply with `WanVideoActivationEditor` (strength 0.3-0.5)
5. Generate with `WanVideoSampler`

### Lightx2v Notes
- 4-step inference amplifies effects
- Use lower strength (0.1-0.2)
- Late blocks work best

## Test Examples (WAN-Style Prompts)

### Environmental Transformation
**Office → Volcanic** (late blocks, strength 0.4)
- Main: "A woman sits at a wooden desk in a modern office, typing on a silver laptop. Behind her, floor-to-ceiling windows reveal a cityscape. The office has minimalist decor with white walls."
- Injection: "Molten lava flows down a volcanic mountainside. The glowing orange-red magma radiates intense heat, causing the air to shimmer. Smoke and steam rise from the flow."
- Result: Office with lava-like lighting, heat distortion effects

**Living Room → Blizzard** (mid blocks, strength 0.5)
- Main: "A family gathers around a stone fireplace in a cozy living room. Children play on the carpet while parents relax on a leather sofa. Warm lighting from table lamps creates intimacy."
- Injection: "A powerful blizzard engulfs a mountain landscape. Snow falls heavily, driven by fierce winds. Visibility is reduced to mere meters as the storm rages."
- Result: Interior with frost patterns, cold blue lighting

### Activity Contrast
**Gardening → DJ Performance** (alternating blocks, strength 0.3)
- Main: "An elderly man tends to his garden in morning light. He kneels beside tomato plants, pruning with small shears. His weathered hands move with practiced precision."
- Injection: "A DJ performs at a packed nightclub. Hands move rapidly across turntables. Strobe lights flash in sync with pounding bass. The crowd dances energetically."
- Result: Surreal mix of peaceful gardening with rhythmic energy

### Material Transformation
**Tech Lab → Rainforest** (early blocks, strength 0.6)
- Main: "A robotics laboratory buzzes with activity. Robotic arms perform assembly tasks. Engineers monitor screens showing data. The space has clean white surfaces."
- Injection: "An ancient rainforest teems with life. Massive trees rise like pillars, covered in moss and vines. Exotic birds call from the canopy. Mist hangs in the air."
- Result: Lab equipment with organic, moss-like textures

### Custom Patterns
- **Bookend**: `1111111111000000000000000000001111111111` (strong start/end)
- **Wave**: `0011110000111100001111000011110000111100` (periodic influence)

## Advanced Techniques

- **Multi-stage**: Chain editors for layered effects
- **Strength ranges**: 0.1-0.3 (subtle), 0.4-0.6 (balanced), 0.7-1.0 (strong)
- **Complementary prompts**: Work best when sharing conceptual elements
- **Compatible with**: NAG, Radial Attention, LoRA

## Troubleshooting

**No visible effect**: 
- Check prompts are >50% different (see debug output)
- Ensure some blocks are activated (not all zeros)
- Try increasing strength to 0.6-0.8
- Verify all connections are made

**Too strong/chaotic**: 
- Reduce strength to 0.1-0.3
- Use fewer active blocks
- Try `sparse` preset instead of continuous blocks

**Checking prompt difference**:
Set log level to "verbose" and look for:
```
[WAN_DEBUG] Block 0 Blending Results:
  - Percent changed: 78.4%  ✓ GOOD - very different
  - Percent changed: 1.1%   ✗ BAD - too similar
```

**Verifying activation**:
```
[WanVideoActivationEditor] Active blocks: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
✓ Model patched successfully - 10 blocks will be modified
```

## Best Practices

### WAN Prompt Style
Write detailed scene descriptions including:
- Subject appearance and actions
- Environment and setting details
- Lighting and atmosphere
- Object positions and relationships
- Colors, textures, materials

### Effective Workflow
1. **Test prompt difference first**: Use WanVideoVectorDifference to verify >50% difference
2. **Start with presets**: `early_blocks` for textures, `late_blocks` for concepts
3. **Initial strength**: 0.3-0.5 (reduce to 0.1-0.2 for Lightx2v)
4. **Use log level**: Set to "basic" or "verbose" to monitor activation

### Performance
- No additional VRAM usage (model not duplicated)
- Minimal computation overhead
- Works with all WanVideo model variants

## Creative Applications

- **Style transfer**: Early blocks
- **Object morphing**: Mid blocks
- **Concept blending**: Late blocks
- **Mood shifts**: All blocks, low strength

## Example Results

**Underwater Office** (late_blocks, 0.4): Office with aquatic lighting, floating papers
**Mechanical Garden** (mid_blocks, 0.5): Flowers with metallic petals
**Time Blend** (custom middle blocks, 0.6): Medieval castle with futuristic elements