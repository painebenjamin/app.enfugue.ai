/** @module controlletr/sidebar/04-refining */
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
                    "value": 0.3
                }
            },
            "refinerGuidanceScale": {
                "label": "Refiner Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 5.0
                }
            },
            "refinerAestheticScore": {
                "label": "Refiner Aesthetic Score",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 6.0
                }
            },
            "refinerNegativeAestheticScore": {
                "label": "Refiner Negative Aesthetic Score",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                    "step": 0.01,
                    "value": 2.5
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
        this.subscribe("modelPickerChange", (newModel) => {
            if (newModel.status.xl) {
                this.refiningForm.show();
            } else {
                this.refiningForm.hide();
            }
        });
    }
}

export { RefiningController as SidebarController };
