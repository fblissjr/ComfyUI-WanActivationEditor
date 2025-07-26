# WanVideo Activation Editor

Experimental block-level activation editing for WanVideo models, along with some tooling and a DuckDB database for storing data points and embeddings. Inject alternative text conditioning into specific transformer blocks to create hybrid semantic effects, control style transfer, and achieve fine-grained control over video generation.

Built with love for the Bandoco community, where all the amazing innovation in video generation is happening right now. Special thanks as always to kijai for the nonstop incredibly hard work on [WanVideoWrapper](https://github.com/kijai/WanVideoWrapper), which this extends and builds upon.

### The Challenge: Features Don't Live in Neat Boxes

Traditional prompting affects all blocks equally, but **neural networks don't organize features the way humans do.** Like LLMs, these things learn representations that get repurposed in ways that are often counterintuitive and difficult to predict without an expensively trained set of interpretability models and tools. For a good example of this, see the recent Anthropic paper on (agent alignment faking)[https://alignment.anthropic.com/2025/automated-auditing/].

During training, models learn representations that are:
- **Distributed**: A single concept like "motion" isn't stored in one block - it's spread across many blocks, attention heads, and neurons
- **Entangled**: Features overlap and interact in non-intuitive ways. "Color" and "texture" might be inseparable in certain layers
- **Hierarchical but messy**: While there's often a progression from low-level to high-level features, it's not a clean separation
- **Task-dependent**: What a block "does" can change based on what you're generating

The common wisdom about "early = simple, late = complex" is an oversimplification. In reality:
- Any given block processes MANY different types of features simultaneously
- Features get transformed and recombined as they flow through the network
- The same block might handle color in one context and motion in another

So why do block-level injections work at all? Because even though features are distributed, **certain blocks tend to be more influential for certain types of changes**, but it's not always clear why without a lot of data points and trial and error and intuition.

This is why we need:
1. **Granular control** (per-block strengths) - because effects aren't uniform
2. **Systematic experimentation** - to map tendencies, not rigid functions
3. **Community collaboration** - more data points = better understanding

See [BLOCK_ANALYSIS.md](BLOCK_ANALYSIS.md) for ideas on mapping these complex relationships.

## What is it?

WanVideo Activation Editor is an experimental tool that attempts to inject different text prompts into specific layers of the WanVideo transformer during generation. Think of it like having 40 different control knobs (one for each transformer block) where you can blend different concepts at different depths of the model.

**What you can do with it:**
- Style transfer (inject artistic style while preserving content)
- Gradual concept morphing through progressive block activation
- Hybrid objects that shouldn't exist ("cyberpunk forest", "liquid architecture")
- Fine-grained control over generation
- Block-by-block experimentation to map transformer behavior

Now the real fun begins - help us discover what each block does!

## Quick Start

1. Install in your ComfyUI `custom_nodes` folder
2. Make sure [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) is installed
3. Restart ComfyUI
4. Add the nodes to your own workflow, or load an example workflow from `example_workflows/`
4a. Basic nodes to add: `WanVideo Block Activation Builder`, `WanVideo Activation Editor`, `WanVideo Block Activation Viewer` if you prefer to visualize the activation patterns in the UI.
4b. Advanced nodes to add: `WanVideo Activation Database`, `WanVideo Activation Vector Operations`, `WanVideo Activation Strength Patterns`
4c. Add two text encoder nodes to the workflow - one for the main prompt embedding and one for the injection prompt embedding.
4d. Connect the `WanVideo Model Loader`'s `model` output to the `WanVideo Activation Editor` node's input of `model`.
4e. Connect the main prompt encoder to the `WanVideo Activation Editor` node's input of `text_embeds`.
4f. Connect the injection prompt encoder to the `WanVideo Activation Editor` node's input of `text_embeds_injection`.
4g. Connect the `WanVideo WanVideo Block Activation Builder` node's output of `block_activations` to the `WanVideo Block Activation Editor` node's input of `block_activations` (you can manually edit the blocks of 1s and 0s if you want as well).
5. Start experimenting

## Example Workflows

Four workflows included:
- **`simple_activation.json`** - Minimal setup to get started with basic injection
- **`amplified_injection.json`** - Shows how to use the Embedding Amplifier for better results
- **`features_showcase.json`** - Demonstrates all features: vector ops, strength patterns, database
- **`debug_workflow.json`** - Full debugging setup for troubleshooting and analysis

## Core Nodes

### Activation Editing

**WanVideoActivationEditor** - The main node for block-level prompt injection
- Injects alternative text conditioning into selected transformer blocks
- Uniform strength control (0.0-1.0) across all active blocks
- 40-bit activation pattern for precise block selection
- Runtime patching without modifying WanVideoWrapper
- Optional pre-projection through text embedding layer for better performance
- Built-in debug output shows patching status and active blocks
- Log level control (off/basic/verbose/trace) for debugging
- **NEW**: Injection mode selection:
  - `context`: Inject into cross-attention context (default, proven to work)
  - `hidden_states`: Inject into block hidden states (experimental)
  - `both`: Inject into both context and hidden states

**WanVideoBlockActivationBuilder** - Visual pattern builder with presets
- Interactive UI for selecting which blocks to activate
- Preset patterns that made sense to me as a starting point. You can of course set your own custom blocks, as well as set individual block strengths for injection embeds. See right below here in `Advanced Strength Control`.
  - `early_blocks`: First 10 blocks (might be low-level features?)
  - `mid_blocks`: Middle 20 blocks (possibly object/scene stuff?)
  - `late_blocks`: Last 10 blocks (maybe details?)
  - `alternating`: Every other block (don't know, seemed useful to toggle as a preset)
  - `sparse`: Every 4th block (subtle... something)
  - Plus `all`, `none`, `first_half`, `second_half`, `custom`
- JavaScript-enhanced UI updates checkboxes when preset changes

**WanVideoBlockActivationViewer** - Debug and visualization
- See current activation configuration from model
- Visual block pattern display (connect block_activations output to see pattern)
- Verify injection is working correctly
- Pattern analysis (detects common patterns)

**WanVideoInjectionTester** - Test injection effectiveness
- Analyzes embedding differences between main and injection prompts
- Shows which dimensions differ most
- Warns if prompts are too similar for effective injection
- Helps debug why injection might not be working

**WanVideoEmbeddingAmplifier** - Simple solution for low embedding differences
- Automatically amplifies differences between embeddings to target percentage
- Three modes: push_apart (gentle), maximize_diff (aggressive), orthogonalize (mathematical)
- Default target of 60% difference ensures visible effects
- Preserves embedding structure while increasing contrast

### Advanced Strength Control

**WanVideoBlockStrengthBuilder** - Create per-block strength patterns with presets (uniform, linear_decay, gaussian_peak, etc.)

**WanVideoAdvancedActivationEditor** - Use per-block strength patterns instead of uniform strength

**WanVideoStrengthVisualizer** - ASCII visualization of strength patterns

### Vector Operations

**WanVideoEmbeddingAnalyzer** - Statistical analysis and dimensionality info with auto-storage

**WanVideoVectorDifference** - Extract concept vectors with `A - B` operations

**WanVideoVectorArithmetic** - Complex math operations on up to 4 embeddings

**WanVideoVectorInterpolation** - Smooth transitions between embeddings (linear/spherical/cubic)

**WanVideoEmbeddingDatabase** - DuckDB storage with compression and SQL queries

## How It Works

### WanVideo +  Architecture

WanVideo uses a 40-block transformer architecture. During generation, text embeddings flow through these blocks, with each block refining the representation. By intercepting and modifying the embeddings at specific blocks, we can sorta control different aspects of generation, or at the very least, make some super bizarre and/or broken videos.

### Data Flow

My professional experience is based on a data background over 20 years (and LLMs and DiT models and diffusion models for the past 3 years), and for better or worse, I tend to think in terms of data flows.

1. **Text Encoding**: Your prompts are encoded into embeddings by the UMT5-XXL text encoder
2. **Dimension Projection**: Raw embeddings (4096-dim) are projected to transformer space (5120-dim)
3. **Block Processing**: As embeddings flow through blocks, our injections blend in at specified points
4. **Controlled Generation**: The modified embeddings guide video generation with your hybrid semantics

## Technical Details

1. **Memory**: Auto CPU offload, garbage collection, 3-4x compression
2. **Dimensions**: Auto shape alignment, smart padding, pre-projection when possible
3. **Database**: DuckDB tracks embeddings, operations, performance metrics
4. **Debug**: Console output is off by default. To enable logging:
   - Easy: Set "Log Level" dropdown to "basic" or "verbose" in WanVideoActivationEditor node
   - Or use environment variables:
   ```bash
   export WAN_ACTIVATION_DEBUG=1    # Basic output
   export WAN_ACTIVATION_VERBOSE=1  # Detailed output
   ```

## (Potentially) Best Practices, But Who Knows?

**Testing**: Single blocks at strength 1.0, document effects, share findings

**Strength**: Start 0.3-0.5 uniform, then try per-block patterns (gradients, peaks)

**Performance**: Cache common embeddings, batch operations

**Creative Examples**:
- Style extraction: `encode("oil painting") - encode("photo")`
- Progressive morph: Linear gradient 0→1 across blocks
- Hybrid concepts: "wolf" + "liquid mercury" = metallic flowing wolf

## The Simple Solution: Embedding Amplifier

If your prompts aren't different enough (common with FP8 quantization), use the **WanVideoEmbeddingAmplifier** node:

1. Connect your main prompt embeddings
2. Connect your injection prompt embeddings to the amplifier
3. Set target difference to 60% (default)
4. Use the amplified output as your injection embeddings

This ensures your embeddings are different enough for visible effects, even with similar prompts!

## Troubleshooting

Check console output for "Percent changed" - needs to be >50% for visible effects!

Common issues:
- **Low embedding difference (<30%)** → Use the Embedding Amplifier node!
- **No visible effect** → Increase strength or use more different prompts (aim for >50% difference)
- **No blocks found** → Update WanVideoWrapper (or I need to update this repo to match WanVideoWrapper's changes)
- **Dimension mismatch** → Handled automatically

## What (I) Still Don't Know

Which blocks control what in what combinations and strengths for different text embeds and models and lengths and steps and... so on. See [BLOCK_ANALYSIS.md](BLOCK_ANALYSIS.md) for testing ideas. DuckDB is bundled for the purpose of your own testing, but also to facilitate sharing data in the future for open source research.

## Requirements

- ComfyUI + [WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- DuckDB (auto-installed)

## License

MIT
