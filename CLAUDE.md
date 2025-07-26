# WanVideo Activation Editor Project

## Code and Writing Style Guidelines

- **No emojis** in code, display names, or documentation
- Keep all naming and display text professional
- Avoid "Enhanced", "Advanced", "Ultimate" type prefixes - use descriptive names instead
- Clean, simple node names that describe what they do
- Stay professional and concise in writing and in code
- Simplify and clean up code where possible
- Be concise in README.md and other documentation except when detail is important. Avoid the 'no shit' parts of READMEs and keep the tone professional and to the point and direct. No hype or hyperbole.

## Overview
This project implements block-level activation editing and vector arithmetic operations for WanVideo models through ComfyUI nodes. It allows injecting alternative text conditioning into specific transformer blocks and performing mathematical operations on embeddings to create novel effects.

## Current Status
After extensive debugging and fixes, the runtime patching system is now fully functional. All transformer blocks are successfully patched and the forward methods are being called during generation.

**What's Working:**
1. Runtime patching of all 40 transformer blocks
2. Forward method interception confirmed during generation
3. Context parameter successfully captured (shape: [1, 512, 5120])
4. Injection embeddings properly prepared and available
5. Block-level activation configuration stored correctly
6. Vector arithmetic operations with automatic shape alignment
7. DuckDB storage with compression (3-4x reduction)
8. Memory-efficient operations preventing VRAM leaks
9. Debug system simplified with console output

**Recent Fixes:**
- Fixed model path discovery (`model.blocks` instead of `transformer_blocks`)
- Fixed forward method signature for WanAttentionBlock
- Added `@torch.compiler.disable` to prevent torch dynamo issues
- Simplified debug system with built-in console output
- Fixed all `verbose_only` parameter errors
- Added runtime log level control in WanVideoActivationEditor node
- Added injection mode selection (context vs hidden states vs both)
- Added WanVideoInjectionTester for debugging effectiveness

**Next Steps:**
Now that patching is confirmed working, users should experiment with:
- Different injection prompts and strengths
- Various block activation patterns
- Document which blocks produce which effects

## Architecture

### Core Systems
- **Block-level injection**: Stores activation configuration in transformer_options for WanVideoWrapper to process
- **Vector arithmetic**: Mathematical operations on T5 embeddings with automatic shape alignment
- **DuckDB storage**: Persistent embedding cache with zstd compression and SQL analytics
- **Memory management**: Context managers and immediate cleanup to prevent VRAM leaks

### Data Flow
1. **Text Encoding**: WanVideoTextEncode creates raw T5 embeddings [seq_len, 4096]
2. **Vector Operations**: Optional arithmetic operations on embeddings
3. **Configuration**: WanVideoActivationEditor adds injection config to model and embeds
4. **Storage**: Embeddings cached in DuckDB with compression
5. **Sampling**: WanVideoSampler passes embeddings to transformer
6. **Processing**: Transformer processes raw embeddings through text_embedding layer
7. **Injection**: Blocks can access config via transformer_options (requires WanVideoWrapper modification)

## Key Components

### Activation Editing Nodes

#### WanVideoActivationEditor
Main node that configures the model and text embeddings for activation injection. 

**Key Features:**
- Runtime log level control (off/basic/verbose/trace)
- Injection mode selection:
  - `context`: Injects into cross-attention context (default, proven working)
  - `hidden_states`: Injects into transformer hidden states (experimental)
  - `both`: Injects into both context and hidden states
- Embedding difference measurement with warnings for low differences

Stores configuration in:
- `model.model_options.transformer_options['wan_activation_editor']`
- `text_embeds['wan_activation_editor']`

#### WanVideoBlockActivationBuilder
Helper node for creating block activation patterns. Provides presets:
- `early_blocks`: First 10 blocks
- `mid_blocks`: Middle 20 blocks
- `late_blocks`: Last 10 blocks
- `alternating`: Every other block
- `sparse`: Every 4th block
- `custom`: Manual selection

#### WanVideoBlockActivationViewer
Debug node with enhanced visualization:
- Shows current activation configuration
- Visual block pattern display
- Pattern analysis and detection

### Vector Arithmetic Nodes

#### WanVideoEmbeddingAnalyzer
Analyzes embedding structure and statistics:
- Shape and dimension information
- Statistical analysis (mean, std, norms)
- Dimensionality estimation (PCA preview)
- Automatic database storage

#### WanVideoVectorDifference
Computes difference between embeddings:
- `A - B` to extract style or concept vectors
- Optional normalization
- Configurable scaling
- Automatic shape alignment

#### WanVideoVectorArithmetic
Performs complex arithmetic operations:
- Operations: add, weighted_sum, multiply, normalize
- Up to 4 operands with individual weights
- Automatic shape alignment
- Operation tracking in database

#### WanVideoVectorInterpolation
Creates smooth transitions between embeddings:
- Methods: linear, spherical, cubic
- Configurable steps (2-20)
- Optional endpoint inclusion
- Stores full interpolation sequence

#### WanVideoEmbeddingDatabase
Database management interface:
- View statistics and storage metrics
- Clean up old embeddings
- List recent operations
- Performance analytics

## Technical Details

### Database Schema
```sql
-- Core tables
embeddings          -- Compressed embedding storage
vector_operations   -- Operation tracking
operation_inputs    -- Input relationships
operation_parameters -- Operation settings
embedding_analysis  -- Cached analysis results
embedding_similarities -- Similarity scores
```

### Memory Management
- Embeddings moved to CPU immediately after operations
- Compressed with zstd (typical 3-4x compression)
- Stored as float16 to reduce size
- VRAM cleared after each operation
- Context managers ensure cleanup

### Dimension Handling
- Raw T5 embeddings: [seq_len, 4096]
- Automatic shape alignment for mismatched sequences
- Padding with zeros for shorter sequences
- Truncation for longer sequences
- Processed context in transformer: [batch, seq_len, 5120]

### UI Enhancement
- **JavaScript Integration**: Added `web/js/wan_activation_editor.js` for dynamic UI updates
- **Preset Auto-Update**: Block checkboxes automatically update when preset is selected
- **Visual Feedback**: Console shows block patterns as █░ visualization
- **Active Count Display**: Node title shows number of active blocks

### Debug Features
- **Log Level Control**: Dropdown in WanVideoActivationEditor for runtime control
  - `off`: No debug output (default)
  - `basic`: Essential information only
  - `verbose`: Detailed operation logs
  - `trace`: Full trace with stack information
- **Easy to Enable**: Just change dropdown from "off" to "basic" or "verbose"
- **Built-in Console Output**: Shows patching status and active blocks based on log level
- **Enhanced Viewer**: Respects global log level for debug information
- **Real-time Feedback**: See which blocks are being called during generation

## Dependencies
- torch>=2.0.0
- numpy>=1.24.0
- duckdb>=0.9.0
- zstandard>=0.22.0
- ComfyUI-WanVideoWrapper (required, must be in adjacent directory)

## Integration Requirements
The system uses runtime patching to intercept transformer block forward() calls. No modifications to WanVideoWrapper are needed - the patching happens at runtime and has been confirmed working with the latest fixes.

## Testing

### Vector Operations Test
```python
# Test difference extraction (WAN-style prompts)
gothic = "A medieval castle standing on a foggy hillside at dusk. The structure features dark stone walls with Gothic arches and pointed towers. Gargoyles perch on the corners of the battlements, their grotesque forms silhouetted against the darkening sky. The castle walls are weathered and moss-covered, showing centuries of age. Dim orange light emanates from narrow windows, suggesting occupied chambers within. The surrounding landscape is shrouded in mist, with bare trees creating eerie shadows."

regular = "A medieval castle positioned on a grassy hillside during daytime. The fortress has sturdy stone walls with rectangular towers at each corner. The battlements are simple and functional, with crenellations for defense. The castle appears well-maintained with clean stonework. Bright sunlight illuminates the structure, creating clear shadows. The surrounding landscape shows green fields and distant forests under a clear blue sky."

style = difference(gothic, regular)

# Apply to new scene
beach = "A tropical beach scene at midday with crystal clear turquoise water. White sand stretches along the shoreline, dotted with several palm trees swaying gently in the breeze. Small waves lap against the shore, creating a rhythmic pattern. The sky is bright blue with a few wispy clouds. In the distance, a small wooden boat is anchored near the shore. The scene conveys a peaceful, vacation atmosphere."

gothic_beach = add(beach, style * 0.5)
```

### Database Verification
1. Generate several embeddings
2. Check database stats node
3. Verify compression ratios
4. Test deduplication

### Performance Monitoring
- Watch VRAM usage during operations
- Check operation timing in database
- Monitor compression effectiveness

## Injection Mode Insights

### Context Injection (Default)
- Modifies the cross-attention context that stays constant across all blocks
- Proven to work and produce visible effects when prompts differ by >50%
- Safe and stable approach

### Hidden State Injection (Experimental)
- Attempts to modify the hidden states (x) that flow through transformer blocks
- More direct intervention in the generation process
- Currently limited by dimension mismatch (T5 4096-dim vs hidden 5120-dim)
- Would require projecting T5 embeddings through initial model layers first

### Why Low Embedding Differences Occur
1. **FP8 Quantization**: Limits embedding diversity during T5 generation
2. **Similar Prompts**: T5 produces similar embeddings for semantically related text
3. **Measurement Location**: We measure raw T5 output, not post-projection embeddings

### Debugging Injection Effectiveness
Use the WanVideoInjectionTester node to:
- Verify embedding differences are >50% for visible effects
- Check which dimensions differ most between prompts
- Confirm patching is active and blocks are being called

## Code Guidelines
- Always modify the main implementation directly
- Use type hints for clarity
- Include debug print statements
- Handle edge cases gracefully
- Clean up resources immediately

## Future Roadmap
See ROADMAP.md for planned features including:
- Control vectors and direction arithmetic
- Ablation techniques for concept removal
- Rotation matrices for semantic transformations
- Multi-modal control (image/audio guided)
- Learned control vectors
