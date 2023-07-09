/** @module controller/sidebar/03-tweaks */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { Controller } from "../base.mjs";
import { 
    NumberInputView, 
    FloatInputView,
    CheckboxInputView
} from "../../view/forms/input.mjs";

let defaultGuidanceScale = 7.5,
    defaultInferenceSteps = 50;

/**
 * The forms that allow for tweak inputs
 */
class TweaksForm extends FormView {
    /**
     * @var bool Don't show submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object The tweak fields
     */
    static fieldSets = {
        "Tweaks": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "value": defaultGuidanceScale,
                    "step": 0.1,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness."
                }
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 5,
                    "max": 250,
                    "value": defaultInferenceSteps,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process but can produce better results."
                }
            }
        }
    };
}

/**
 * Extend the menu controll to bind init
 */
class TweaksController extends Controller {
    /**
     * Return data from the tweaks form
     */
    getState() {
        return { "tweaks": this.tweaksForm.values };
    }

    /**
     * Sets state in the form
     */
    setState(newState) {
        if (!isEmpty(newState.tweaks)) {
            this.tweaksForm.setValues(newState.tweaks).then(() => this.tweaksForm.submit());
        }
    }

    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "tweaks": {
                "guidanceScale": defaultGuidanceScale,
                "inferenceSteps": defaultInferenceSteps
            }
        }
    }

    /**
     * On initialization, append the Tweaks form
     */
    async initialize() {
        // Set defaults
        defaultGuidanceScale = this.application.config.model.invocation.guidanceScale;
        defaultInferenceSteps = this.application.config.model.invocation.inferenceSteps;
        
        // Builds form
        this.tweaksForm = new TweaksForm(this.config);
        this.tweaksForm.onSubmit(async (values) => {
            this.engine.guidanceScale = values.guidanceScale;
            this.engine.inferenceSteps = values.inferenceSteps;
        });

        // Add to sidebar
        this.application.sidebar.addChild(this.tweaksForm);
    }
}

export { TweaksController as SidebarController }
