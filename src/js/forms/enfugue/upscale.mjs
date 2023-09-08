/** @module forms/enfugue/upscale */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import { 
    CheckboxInputView,
    NumberInputView,
    MaskTypeInputView,
    SchedulerInputView,
    FormInputView,
    RepeatableInputView,
    UpscaleAmountInputView,
    UpscaleMethodInputView,
    UpscaleDiffusionControlnetInputView,
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
     * @var object All field sets and their config
     */
    static fieldSets = {
        "Upscaling Step": {
            "amount": {
                "label": "Upscale Amount",
                "class": UpscaleAmountInputView
            },
            "method": {
                "label": "Upscale Method",
                "class": UpscaleMethodInputView
            },
            "strength": {
                "label": "Denoising Strength",
                "class": UpscaleDiffusionStrengthInputView
            },
            "pipeline": {
                "label": "Pipeline",
                "class": UpscaleDiffusionPipelineInputView
            },
            "controlnet": {
                "label": "ControlNet",
                "class": UpscaleDiffusionControlnetInputView
            },
            "prompt": {
                "label": "Detail Prompt",
                "class": UpscaleDiffusionPromptInputView
            },
            "negativePrompt": {
                "label": "Detail Negative Prompt",
                "class": UpscaleDiffusionNegativePromptInputView
            },
            "scheduler": {
                "label": "Scheduler",
                "class": SchedulerInputView
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": UpscaleDiffusionStepsInputView
            },
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": UpscaleDiffusionGuidanceScaleInputView
            },
            "chunkingSize": {
                "label": "Chunking Size",
                "class": NumberInputView,
                "config": {
                    "minimum": 32,
                    "maximum": 512,
                    "step": 8,
                    "value": 128,
                    "tooltip": "The number of pixels to move the frame by during diffusion. Smaller values produce better results, but take longer."
                }
            },
            "chunkingMaskType": {
                "label": "Chunking Mask",
                "class": MaskTypeInputView,
            }
        }
    };

    /**
     * @var object The conditions to display some inputs
     */
    static fieldSetConditions = {
        "Upscale Diffusion": (values) => values.diffusion
    };
}

/**
 * Extend the class to enable auto submission
 */
class AutoSubmitUpscaleFormView extends UpscaleFormView {
    /**
     * @var bool Enable autosubmit
     */
    static autoSubmit = true;
}

/**
 * Extend the form input view class to hold the whole upscale form
 */
class UpscaleFormInputView extends FormInputView {
    /**
     * @var class The form view
     */
    static formClass = AutoSubmitUpscaleFormView;
}

/**
 * Extend the repetable input view to allow multiple upscale forms
 */
class UpscaleStepsInputView extends RepeatableInputView {
    /**
     * @var string The no item label
     */
    static noItemsLabel = "Press `+` to add an upscaling step";

    /**
     * @var class The form input view
     */
    static memberClass = UpscaleFormInputView;
}

/**
 * Create a form view for the sidebar that adds multiple upscale steps
 */
class UpscaleStepsFormView extends FormView {
    /**
     * @var bool Autosubmit
     */
    static autoSubmit = true;

    /**
     * @var bool Hide fieldsets
     */
    static collapseFieldSets = true;

    /**
     * @var object The upscale steps
     */
    static fieldSets = {
        "Upscaling": {
            "steps": {
                "class": UpscaleStepsInputView
            }
        }
    };
}

/**
 * The quick downscale form is used when a user directly selects 'Downscale' from an image
 */
class DownscaleFormView extends FormView {
    /**
     * @var bool Show the cancel button
     */
    static showCancel = true;

    /**
     * @var object Just one fieldset
     */
    static fieldSets = {
        "Downscale Amount": {
            "downscale": {
                "class": UpscaleAmountInputView,
                "config": {
                    "tooltip": "Select the amount of downscaling to apply. Downscaling is performed via repeated bi-linear sampling."
                }
            }
        }
    }
};

export {
    UpscaleFormView,
    UpscaleStepsFormView,
    DownscaleFormView
};
