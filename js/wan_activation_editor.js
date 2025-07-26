import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "WanVideo.ActivationEditor",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WanVideoBlockActivationBuilder") {
            console.log("[WanVideo] Registering BlockActivationBuilder extensions");
            
            // Override onNodeCreated to add our functionality
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                console.log("[WanVideo] BlockActivationBuilder node created");
                setupPresetSync(this);
                
                return result;
            };
        }
    },
    
    async nodeCreated(node) {
        // Additional setup when any node is created
        if (node.type === "WanVideoBlockActivationBuilder") {
            // Ensure preset sync is set up (in case it wasn't in beforeRegisterNodeDef)
            setTimeout(() => {
                setupPresetSync(node);
            }, 100);
        }
    }
});

function setupPresetSync(node) {
    // Find the preset widget
    const presetWidget = node.widgets?.find(w => w.name === "preset");
    if (!presetWidget) {
        console.warn("[WanVideo] Preset widget not found on node", node.id);
        return;
    }
    
    // Don't set up twice
    if (presetWidget._wanVideoSetup) {
        return;
    }
    presetWidget._wanVideoSetup = true;
    
    console.log("[WanVideo] Setting up preset sync for node", node.id);
    
    // Inject into the existing callback
    const originalCallback = presetWidget.callback;
    presetWidget.callback = function(value, app, node, pos, event) {
        console.log(`[WanVideo] Preset changed to: ${value}`);
        
        // Update blocks first
        updateBlocksForPreset(node, value);
        
        // Then call original callback
        if (originalCallback) {
            originalCallback.call(this, value, app, node, pos, event);
        }
        
        // Force redraw
        node.setDirtyCanvas(true);
        app.graph?.setDirtyCanvas(true);
    };
    
    // If preset already has a value, apply it now
    if (presetWidget.value && presetWidget.value !== "custom") {
        console.log(`[WanVideo] Applying initial preset: ${presetWidget.value}`);
        updateBlocksForPreset(node, presetWidget.value);
    }
}

function updateBlocksForPreset(node, preset) {
    if (preset === "custom") {
        console.log("[WanVideo] Custom preset - not changing blocks");
        return;
    }
    
    console.log(`[WanVideo] Updating blocks for preset: ${preset}`);
    
    const NUM_BLOCKS = 40;
    let pattern = new Array(NUM_BLOCKS).fill(false);
    
    // Define patterns
    switch(preset) {
        case "all":
            pattern.fill(true);
            break;
            
        case "none":
            pattern.fill(false);
            break;
            
        case "first_half":
            for (let i = 0; i < NUM_BLOCKS / 2; i++) {
                pattern[i] = true;
            }
            break;
            
        case "second_half":
            for (let i = NUM_BLOCKS / 2; i < NUM_BLOCKS; i++) {
                pattern[i] = true;
            }
            break;
            
        case "early_blocks":
            for (let i = 0; i < 10; i++) {
                pattern[i] = true;
            }
            break;
            
        case "mid_blocks":
            for (let i = 10; i < 30; i++) {
                pattern[i] = true;
            }
            break;
            
        case "late_blocks":
            for (let i = 30; i < 40; i++) {
                pattern[i] = true;
            }
            break;
            
        case "alternating":
            for (let i = 0; i < NUM_BLOCKS; i += 2) {
                pattern[i] = true;
            }
            break;
            
        case "sparse":
            for (let i = 0; i < NUM_BLOCKS; i += 4) {
                pattern[i] = true;
            }
            break;
    }
    
    // Apply pattern to widgets
    let updatedCount = 0;
    for (let i = 0; i < NUM_BLOCKS; i++) {
        const blockWidget = node.widgets?.find(w => w.name === `block_${i}`);
        if (blockWidget) {
            blockWidget.value = pattern[i];
            updatedCount++;
            
            // Call the widget's callback if it exists
            if (blockWidget.callback) {
                blockWidget.callback(blockWidget.value, app, node);
            }
        }
    }
    
    console.log(`[WanVideo] Updated ${updatedCount}/${NUM_BLOCKS} block widgets`);
    
    // Update node title
    const activeCount = pattern.filter(v => v).length;
    node.title = `Block Activation Builder (${activeCount} active)`;
    
    // Visual pattern for debugging
    let visual = "";
    for (let i = 0; i < NUM_BLOCKS; i++) {
        if (i > 0 && i % 10 === 0) visual += " ";
        visual += pattern[i] ? "█" : "░";
    }
    console.log(`[WanVideo] Pattern: ${visual}`);
}