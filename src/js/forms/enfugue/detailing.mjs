/** @module forms/enfugue/denoising */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import {
    SliderPreciseInputView,
    CheckboxInputView,
    NumberInputView,
    UpscaleDiffusionControlnetInputView,
} from "../input.mjs";

/**
 * The form class containing options for after-details
 */
class DetailingFormView extends FormView {
    /**
     * @var object All field sets and their config
     */
    static fieldSets = {
        "Detailer": {
            "faceRestore": {
                "label": "Restore Faces",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, faces will be identified in the image and restored using GFPGAN. This will improve the appearance of human faces, but can sometimes look out-of-place. This is best used when combined with an additional Face Fix pass."
                }
            },
            "faceInpaint": {
                "label": "Inpaint Faces",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, faces will be identified in the image and inpainted using the current model. This will help make faces look more natural. When used with face restore, this will be performed after."
                }
            },
            "handInpaint": {
                "label": "Inpaint Hands",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, hands will be identified in the image and inpainted using the current model. This can help with poorly formed human hands, but is not guaranteed."
                }
            },
            "inpaintStrength": {
                "label": "Inpaint Denoising Strength",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.25,
                    "step": 0.01,
                    "tooltip": "The strength of the face and/or hand fix denoising pass. Higher values will change the face more from the initial generation."
                }
            },
            "detailStrength": {
                "label": "Detailer Denoising Strength",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.0,
                    "step": 0.01,
                    "tooltip": "When this value is greater than 0, a final detailer pass will be performed on the image with this denoising strength after inpainting faces/hands."
                }
            },
            "detailControlnet": {
                "label": "Detailer ControlNet",
                "class": UpscaleDiffusionControlnetInputView,
                "config": {
                    "tooltip": "Optional, the ControlNet to use when detailing."
                }
            },
            "detailControlnetScale": {
                "label": "Detail ControlNet Scale",
                "class": NumberInputView,
                "config": {
                    "min": 0.0,
                    "value": 1.0,
                    "tooltip": "How closely to follow the ControlNet's guidance. When there is no ControlNet, this has no effect."
                }
            },
            "detailGuidanceScale": {
                "label": "Detailer Guidance Scale",
                "class": NumberInputView,
                "config": {
                    "min": 0.0,
                    "step": 0.01,
                    "tooltip": "The guidance scale to use when detailing. When blank, the guidance scale of the initial inference pass will be used."
                }
            },
            "detailInferenceSteps": {
                "label": "Detailer Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "step": 1,
                    "tooltipo": "The number of inference steps to use when detailing. When blank, the number of inference steps of the initial execution will be used."
                }
            }
        }
    };

    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;

    /**
     * @var bool Collapsed
     */
    static collapseFieldSets = true;
};

export {
    DetailingFormView
};
