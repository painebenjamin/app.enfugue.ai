/** @module controlletr/sidebar/05-refining */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { NumberInputView, FloatInputView } from "../../view/forms/input.mjs";
import { Controller } from "../base.mjs";

/**
 * The RefiningForm gathers inputs for SDXL refining
 */
class RefiningForm extends FormView {
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

/**
 * Extends the menu controller for state and init
 */
class RefiningController extends Controller {
    /**
     * Get data from the refining form
     */
    getState() {
        return { "refining": this.refiningForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "refining": {
                "refinerStrength": 0.3,
                "refinerGuidanceScale": 5.0,
                "refinerAestheticScore": 6.0,
                "refinerNegativeAestheticScore": 2.5
            }
        };
    }

    /**
     * Set state in the refining form
     */
    setState(newState) {
        if (!isEmpty(newState.refining)) {
            this.refiningForm.setValues(newState.refining).then(() => this.refiningForm.submit());
        }
    };

    /**
     * On init, append form and hide until SDXL gets selected
     */
    async initialize() {
        this.refiningForm = new RefiningForm(this.config);
        this.refiningForm.onSubmit(async (values) => {
            this.engine.refinerStrength = values.refinerStrength;
            this.engine.refinerGuidanceScale = values.refinerGuidanceScale;
            this.engine.refinerAestheticScore = values.refinerAestheticScore;
            this.engine.refinerNegativeAestheticScore = values.refinerNegativeAestheticScore;
        });
        this.refiningForm.hide();
        this.application.sidebar.addChild(this.refiningForm);
        this.subscribe("modelPickerChange", (model) => {
            if (isEmpty(model) || isEmpty(model.refiner)) {
                this.refiningForm.hide();
            } else {
                let defaultConfig = model.defaultConfiguration,
                    refiningConfig = {};
                
                if (!isEmpty(defaultConfig.refiner_strength)) {
                    refiningConfig.refinerStrength = defaultConfig.refiner_strength;
                }
                if (!isEmpty(defaultConfig.refiner_guidance_scale)) {
                    refiningConfig.refinerGuidanceScale = defaultConfig.refiner_guidance_scale;
                }
                if (!isEmpty(defaultConfig.refiner_aesthetic_score)) {
                    refiningConfig.refinerAestheticScore = defaultConfig.refiner_aesthetic_score;
                }
                if (!isEmpty(defaultConfig.refiner_negative_aesthetic_score)) {
                    refiningConfig.refinerNegativeAestheticScore = defaultConfig.refiner_negative_aesthetic_score;
                }
                if (!isEmpty(refiningConfig)) {
                    this.refiningForm.setValues(refiningConfig);
                }
                this.refiningForm.show();
            }
        });
        this.subscribe("engineRefinerChange", (refiner) => {
            if (isEmpty(refiner)) {
                this.refiningForm.hide();
            } else {
                this.refiningForm.show();
            }
        });
    }
}

export { RefiningController as SidebarController };
