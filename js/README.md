# WanVideo Activation Editor JavaScript

This directory contains the client-side JavaScript for the WanVideo Activation Editor ComfyUI nodes.

## Files

- `wan_activation_editor.js` - Main extension that enhances the Block Activation Builder node with dynamic preset functionality

## Features

### Block Activation Builder Enhancement

The JavaScript extension adds dynamic behavior to the WanVideoBlockActivationBuilder node:

1. **Preset Auto-Update**: When a preset is selected from the dropdown, all 40 block checkboxes automatically update to match the preset pattern
2. **Visual Feedback**: The node title shows the count of active blocks (e.g., "Block Activation Builder (10 active)")
3. **Preset Patterns**:
   - `all` - Activates all 40 blocks
   - `none` - Deactivates all blocks
   - `first_half` - Activates blocks 0-19
   - `second_half` - Activates blocks 20-39
   - `early_blocks` - Activates blocks 0-9
   - `mid_blocks` - Activates blocks 10-29
   - `late_blocks` - Activates blocks 30-39
   - `alternating` - Activates even-numbered blocks
   - `sparse` - Activates every 4th block
   - `custom` - No auto-update, allows manual selection

## Technical Details

The extension hooks into ComfyUI's node system using:
- `beforeRegisterNodeDef` - To modify node behavior during registration
- Property overrides on the preset widget to intercept value changes
- Widget value updates with proper callbacks to ensure UI synchronization

## Debug Mode

To enable debug output from the activation system:
```bash
export WAN_ACTIVATION_DEBUG=1  # Basic debug messages
export WAN_ACTIVATION_VERBOSE=1  # Detailed debug messages
```