# WanVideo Transformer Block Analysis

## Overview

WanVideo uses a 40-block transformer architecture with cross-attention mechanisms that embed text conditioning into the video generation process. While we don't have exact mappings of what each specific block does, research on transformer architectures provides general patterns.

## General Transformer Block Patterns

Based on transformer architecture research, blocks typically evolve from low-level to high-level processing, though is not guaranteed to follow this pattern.

### Early Blocks (0-13) - Low-Level Features
- **Focus**: Local context and basic patterns
- **Typical processing**:
  - Basic spatiotemporal patterns
  - Local motion detection
  - Color and texture information
  - Edge detection and simple shapes
  - Adjacent frame relationships
- **Attention patterns**: Primarily local, focusing on nearby tokens/pixels

### Middle Blocks (14-26) - Intermediate Features
- **Focus**: Object-level and scene structure
- **Typical processing**:
  - Object boundaries and shapes
  - Phrase-level relationships in text
  - Scene composition
  - Motion trajectories
  - Intermediate semantic features
- **Attention patterns**: Distributed across phrases/regions

### Late Blocks (27-39) - High-Level Features
- **Focus**: Semantic understanding and global coherence
- **Typical processing**:
  - Complex semantic relationships
  - Global scene understanding
  - Long-range dependencies
  - Style and aesthetic qualities
  - Task-specific adaptations
- **Attention patterns**: Sparse, long-range, semantically driven

## WanVideo-Specific Architecture

### Key Components per Block:
1. **Cross-Attention Layer**: Embeds T5 text encodings into the visual processing
2. **Time Embedding MLP**: Shared across blocks with unique biases per block
3. **Self-Attention**: Processes spatiotemporal relationships
4. **Feed-Forward Network**: Independent token processing

### Important Caveats

**These are general patterns, not rigid rules!** In reality:
- Features are distributed across multiple blocks
- The same block might handle different features in different contexts
- There's significant overlap between block functions
- Learned representations are entangled and non-interpretable

## Critical Requirement: Prompt Difference

**Before any testing**: Ensure your main and injection prompts produce >50% different embeddings. The UMT5-XXL encoder creates similar embeddings for semantically related prompts, making effects invisible if prompts are too similar.

Use WanVideoVectorDifference to verify prompt difference before testing block patterns.

## Experimental Approach

To map what blocks actually do in WanVideo:

### 1. Single Block Activation
Test each block individually at high strength (0.9-1.0) with very different prompts:
```
Main: Detailed scene description (WAN style - full paragraph)
Injection: Completely different domain/concept/style
Block 0 only: Note color, texture, basic pattern changes
Block 10 only: Look for object-level modifications
Block 20 only: Check for semantic alterations
Block 30 only: Observe style/mood transformations
Block 39 only: Look for final detail refinements
```

### 2. Progressive Activation
Activate blocks progressively:
```
Blocks 0-5: Early features only
Blocks 0-10: Add more complexity
Blocks 0-20: Include mid-level features
Blocks 0-30: Nearly complete
All blocks: Full processing
```

### 3. Selective Patterns
Test specific hypotheses:
```
Even blocks only: See what's missing
Odd blocks only: Compare differences
First quarter (0-9): Basic features
Last quarter (30-39): Refinements
Middle only (10-29): Core processing
```

### 4. Strength Gradients
Use BlockStrengthBuilder patterns:
```
Linear decay: Strong early → weak late
Linear rise: Weak early → strong late
Gaussian peak at 20: Focus on middle
Random: Chaos testing
```

## Documentation Template

When testing, document:
1. **Prompts**: Full WAN-style / ShareGPT4V descriptions for both main and injection
2. **Embedding Difference**: Percentage from debug output or VectorDifference node
3. **Setup**: Which blocks activated, injection strength used
4. **Observations**: Specific visual changes noticed
5. **Hypothesis**: What you think the blocks control
6. **Reproducibility**: Can you get consistent results?

### Example Entry:
```
Test: Environmental mood transfer
Main: "A busy cafe interior during morning rush. Baristas prepare coffee behind a marble
      counter. Customers wait in line, checking phones. Natural light streams through
      large windows. The space has industrial decor with exposed brick walls."

Injection: "A serene zen garden at dawn. Carefully raked gravel patterns surround
           smooth stones. A small fountain creates gentle water sounds. Morning mist
           drifts through bamboo. The atmosphere is profoundly peaceful."

Embedding difference: 72%
Blocks: 25-35 (late-mid to early-late)
Strength: 0.5
Result: Cafe maintained structure but gained zen-like calm, muted colors, softer lighting
Hypothesis: These blocks influence mood/atmosphere while preserving scene structure
```

## Current Understanding

Based on the architecture and general transformer patterns:
- **Blocks 0-10**: Likely handle basic visual features, motion, color
- **Blocks 10-30**: Probably process objects, scene composition, relationships
- **Blocks 30-40**: Possibly refine details, ensure coherence, apply style

**Remember**: These are tendencies, not rules. Actual behavior depends on:
- The specific prompts used (must be very different!)
- Model version and training
- Interaction effects between blocks
- The inherent entanglement of learned features

## Testing Recommendations

1. **Always verify prompt difference first** - This is the #1 cause of "no effect"
2. **Use WAN-style detailed prompts** - The model was trained on descriptive captions
3. **Start with extreme opposites** - Photo vs painting, calm vs chaotic, indoor vs outdoor
4. **Document percentage differences** - Helps others reproduce your findings
5. **Share your discoveries** - Community mapping will reveal patterns faster
