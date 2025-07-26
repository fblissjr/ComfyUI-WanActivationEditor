# Experimental Directory

This directory contains experimental code, examples, and approaches that aren't currently used in the main nodes but might be useful for future development.

## Contents

### injection_methods.py
Alternative approaches to activation injection that work at different levels:
- **LatentSpaceInjector**: Manipulates latent space between denoising steps
- **CrossAttentionHijacker**: Intercepts cross-attention layers
- **GuidanceModulator**: Modifies classifier-free guidance
- **NoiseScheduleManipulator**: Adjusts noise schedules per block
- **FeatureMapBlender**: Caches and blends intermediate features
- **SamplerInterceptor**: Wraps the sampler for injection

These methods represent different strategies for achieving similar effects to block-level activation editing. They're not currently integrated but provide alternative approaches if the main method has limitations.

### wanvideo_activation_patch.py
Example code showing how to manually patch WanVideoWrapper to support activation editing. This demonstrates:
- Where to add activation support in the model code
- How to read configuration from transformer_options
- The blending logic for context injection

This is useful if you want to:
- Understand how the activation system would work natively
- Manually patch WanVideoWrapper yourself
- Implement similar functionality in other models

## Status
These files are not imported or used by the main nodes. They're kept for:
- Reference and documentation
- Future development possibilities
- Understanding alternative approaches
- Educational purposes