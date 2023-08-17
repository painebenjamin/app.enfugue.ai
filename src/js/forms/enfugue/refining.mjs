/** @module forms/enfugue/refining */
import { FormView } from "../base.mjs";
import { NumberInputView, FloatInputView } from "../input.mjs";

/**
 * The RefiningFormView gathers inputs for SDXL refining
 */
class RefiningFormView extends FormView {
    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object All the inputs in this controller
     */
    static fieldSets = {
        "Refining": {
            "refinerStrength": {
                "label": "Refiner Denoising Strength",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                    "value": 0.3,
                    "tooltip": "When using a refiner, this will control how much of the original image is kept, and how much of it is replaced with refined content. A value of 1.0 represents total destruction of the first image."
                }
            },
            "refinerGuidanceScale": {
                "label": "Refiner Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 5.0,
                    "tooltip": "When using a refiner, this will control how closely to follow the guidance of the model. Low values can result in soft details, whereas high values can result in high-contrast ones."
                }
            },
            "refinerAestheticScore": {
                "label": "Refiner Aesthetic Score",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 6.0,
                    "tooltip": "Aesthetic scores are assigned to images in SDXL refinement; this controls the positive score."
                }
            },
            "refinerNegativeAestheticScore": {
                "label": "Refiner Negative Aesthetic Score",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 2.5,
                    "tooltip": "Aesthetic scores are assigned to images in SDXL refinement; this controls the negative score."
                }
            }
        }
    };
}

export { RefiningFormView };
