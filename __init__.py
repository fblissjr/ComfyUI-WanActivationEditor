from .nodes import WanVideoActivationEditor, WanVideoBlockActivationBuilder, WanVideoActivationScheduler

NODE_CLASS_MAPPINGS = {
    "WanVideoActivationEditor": WanVideoActivationEditor,
    "WanVideoBlockActivationBuilder": WanVideoBlockActivationBuilder,
    "WanVideoActivationScheduler": WanVideoActivationScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoActivationEditor": "WanVideo Activation Editor",
    "WanVideoBlockActivationBuilder": "WanVideo Block Activation Builder", 
    "WanVideoActivationScheduler": "WanVideo Activation Scheduler",
}
