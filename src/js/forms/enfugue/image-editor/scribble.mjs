/** @module forms/enfugue/image-editor/scribble */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../../forms/base.mjs";
import {
    PromptInputView,
    FloatInputView,
    NumberInputView,
    CheckboxInputView,
    SliderPreciseInputView
} from "../../../forms/input.mjs";

class ImageEditorScribbleNodeOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
        "ControlNet Parameters": {
            "conditioningScale": {
                "label": "Conditioning Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "How closely to follow the Scribble ControlNet's influence. Typical values vary, usually values between 0.5 and 1.0 produce good conditioning with balanced randomness, but other values may produce something closer to the desired result."
                }
            },
            "conditioningStart": {
                "label": "Conditioning Start",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.0,
                    "tooltip": "When to begin using the Scribble ControlNet for influence. Defaults to the beginning of generation."
                }
            },
            "conditioningEnd": {
                "label": "Conditioning End",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "When to stop using the Scribble ControlNet for influence. Defaults to the end of generation."
                }
            },
        },
        "Other": {
            "scaleToModelSize": {
                "label": "Scale to Model Size",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When this node has any dimension smaller than the size of the configured model, scale it up so it's smallest dimension is the same size as the model, then scale it down after diffusion.<br />This generally improves image quality in slightly rectangular shapes or square shapes smaller than the engine size, but can also result in ghosting and increased processing time.<br />This will have no effect if your node is larger than the model size in all dimensions.<br />If unchecked and your node is smaller than the model size, TensorRT will be disabled for this node."
                },
            },
            "removeBackground": {
                "label": "Remove Background",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "After diffusion, run the resulting image though an AI background removal algorithm. This can improve image consistency when using multiple nodes."
                }
            }
        },
        "Global Prompt Overrides": {
            "prompt": {
                "label": "Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global prompt will be used."
                }
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global negative prompt will be used."
                }
            },
        },
        "Global Tweaks Overrides": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": null,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness. When left blank, the global guidance scale will be used."
                }
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 5,
                    "max": 250,
                    "step": 1,
                    "value": null,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process. When left blank, the global inference steps will be used."
                }
            },
        }
    };
    
    /**
     * Collapse override fields
     */
    static collapseFieldSets = ["Global Prompt Overrides", "Global Tweaks Overrides"];

    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "options-form-view";
};

export { ImageEditorScribbleNodeOptionsFormView };
