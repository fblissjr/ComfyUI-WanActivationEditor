# WanVideo Data Flow Explorer

A technical web application for exploring the internal data flow and architecture of the WanVideoWrapper and Wan2.1 model(s). Built with vanilla HTML, Tailwind CSS, and JavaScript - no heavy frameworks required.

## Features

- **Pipeline Overview**: Interactive visualization of the complete data flow from text input to video output
- **Text Encoding Analysis**: Real-time token counting and T5-XXL embedding visualization
- **Latent Space Explorer**: Understand compression ratios and patchification process
- **Transformer Architecture**: Somewhat superficial but helpful dive into 40 transformer blocks with specialization curves
- **Memory Analysis**: Detailed breakdown of VRAM usage with optimization suggestions
- **Activation Editor Interface**: Visual block selection and blending analysis

## Running the App

```bash
# From the tools directory
python run_dataflow_webapp.py

# Or directly
python3 run_dataflow_webapp.py
```

The app will automatically open in your default browser at `http://localhost:8080`

## Technical Accuracy

This webapp is based on the actual implementation details from ComfyUI-WanVideoWrapper:

## Requirements

- Python 3.x (uses built-in HTTP server)
- Modern web browser with JavaScript enabled
- No additional Python packages needed

## Architecture Details

The webapp accurately represents:
1. Text encoding through UMT5-XXL
2. Latent space compression and patchification
3. Transformer block processing with self-attention and cross-attention
4. VAE decoding to RGB frames
5. Memory requirements for different configurations

## Configuration Options

- Video dimensions (width, height, frames)
- Batch size
- Model precision (FP32, FP16, FP8)
- Attention modes (SDPA, Flash Attention, SageAttn, Radial Sparse)

All calculations update in real-time as you adjust the parameters.
