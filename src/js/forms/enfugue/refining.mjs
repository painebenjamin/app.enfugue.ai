/** @module forms/enfugue/refining */
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    FloatInputView,
    PromptInputView,
    SliderPreciseInputView
} from "../input.mjs";

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
            "refinerStart": {
                "label": "Refiner Start",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.85,
                    "tooltip": "When using a refiner, this will control at what point during image generation we should switch to the refiner.<br /><br />For example, if you are using 40 inference steps and this value is 0.5, 20 steps will be performed on the base pipeline, and 20 steps performed on the refiner pipeline. A value of exactly 0 or 1 will make refining it's own step, and instead you can use the 'refining strength' field to control how strong the refinement is."
                }
            },
            "refinerStrength": {
                "label": "Refiner Denoising Strength",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                    "value": 0.3,
                    "tooltip": "When using a refiner, this will control how much of the original image is kept, and how much of it is replaced with refined content. A value of 1.0 represents total destruction of the first image. This only applies when using refining as it's own distinct step, e.g., the 'refiner start' value is 0 or 1."
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
            },
            "refinerPrompt": {
                "label": "Refiner Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "The prompt to use during refining. By default, the global prompt will be used."
                }
            },
            "refinerNegativePrompt": {
                "label": "Refiner Negative Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "The negative prompt to use during refining. By default, the global negative prompt will be used."
                }
            }
        }
    };

    /**
     * On submit, disable/enable strength
     */
    async submit() {
        await super.submit();
        if (this.values.refinerStart === 0 || this.values.refinerStart === 1) {
            (await this.getInputView("refinerStrength")).enable();
        } else {
            (await this.getInputView("refinerStrength")).disable();
        }
    }

    /**
     * On first build, trigger submit once
     */
    async build() {
        let node = await super.build();
        return node;
    }
}

export { RefiningFormView };
