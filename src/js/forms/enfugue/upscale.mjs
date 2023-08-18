/** @module forms/enfugue/upscale */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import { 
    CheckboxInputView,
    NumberInputView,
    OutputScaleInputView,
    UpscaleMethodsInputView,
    UpscaleDiffusionIterativeControlnetInputView,
    UpscaleDiffusionPromptInputView,
    UpscaleDiffusionNegativePromptInputView,
    UpscaleDiffusionStepsInputView,
    UpscaleDiffusionStrengthInputView,
    UpscaleDiffusionPipelineInputView,
    UpscaleDiffusionGuidanceScaleInputView
} from "../input.mjs";

/**
 * The form class containing all the above
 */
class UpscaleFormView extends FormView {
    /**
     * @var bool autosubmit when changes are made
     */
    static autoSubmit = true;

    /**
     * @var bool This set is collapsed by default
     */
    static collapseFieldSets = true;
    
    /**
     * @var object All field sets and their config
     */
    static fieldSets = {
        "Upscaling": {
            "outscale": {
                "label": "Output Scale",
                "class": OutputScaleInputView
            }
        },
        "Upscale Methods": {
            "upscale": {
                "label": "Upscale Methods",
                "class": UpscaleMethodsInputView
            },
            "upscaleIterative": {
                "label": "Use Iterative Upscaling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Instead of directly upscaling to the target amount, double in size repeatedly until the image reaches the target size. For example, when this is checked and the upscale amount is 4×, there will be two upscale steps, 8× will be three, and 16× will be four."
                }
            },
            "upscaleDiffusion": {
                "label": "Diffuse Upscaled Samples",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "After upscaling the image use the the algorithm chosen above, use the image as input to another invocation of Stable Diffusion."
                }
            }
        },
        "Upscale Diffusion": {
            "upcsaleDiffusionPipeline": {
                "label": "Pipeline",
                "class": UpscaleDiffusionPipelineInputView
            },
            "upscaleDiffusionControlnet": {
                "label": "ControlNet",
                "class": UpscaleDiffusionIterativeControlnetInputView
            },
            "upscaleDiffusionPrompt": {
                "label": "Detail Prompt",
                "class": UpscaleDiffusionPromptInputView
            },
            "upscaleDiffusionNegativePrompt": {
                "label": "Detail Negative Prompt",
                "class": UpscaleDiffusionNegativePromptInputView
            },
            "upscaleDiffusionSteps": {
                "label": "Inference Steps",
                "class": UpscaleDiffusionStepsInputView
            },
            "upscaleDiffusionStrength": {
                "label": "Denoising Strength",
                "class": UpscaleDiffusionStrengthInputView
            },
            "upscaleDiffusionGuidanceScale": {
                "label": "Guidance Scale",
                "class": UpscaleDiffusionGuidanceScaleInputView
            },
            "upscaleDiffusionChunkingSize": {
                "label": "Chunk Size",
                "class": NumberInputView,
                "config": {
                    "minimum": 32,
                    "maximum": 512,
                    "step": 8,
                    "value": 128,
                    "tooltip": "The number of pixels to move the frame by during diffusion. Smaller values produce better results, but take longer."
                }
            },
            "upscaleDiffusionChunkingBlur": {
                "label": "Chunk Blur",
                "class": NumberInputView,
                "config": {
                    "minimum": 32,
                    "maximum": 512,
                    "step": 8,
                    "value": 128,
                    "tooltip": "The number of pixels to feather the edges of the frame by during diffusion. Smaller values result in more pronounced lines, and large values result in a smoother overall image."
                }
            },
            "upscaleDiffusionScaleChunkingSize": {
                "label": "Scale Chunk Size with Iteration",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "Scale the chunking size ×2 with each iteration of upscaling, with a maximum size of ½ the size of the model."
                }
            },
            "upscaleDiffusionScaleChunkingBlur": {
                "label": "Scale Chunk Blur with Iteration",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "Scale the chunking blur ×2 with each iteration of upscaling, with a maximum size of ½ the size of the model."
                }
            }
        }
    };

    /**
     * @var object The conditions to display some inputs
     */
    static fieldSetConditions = {
        "Upscale Methods": (values) => parseInt(values.outscale) > 1,
        "Upscale Diffusion": (values) => parseInt(values.outscale) > 1 && values.upscaleDiffusion
    };
}

export { UpscaleFormView };
