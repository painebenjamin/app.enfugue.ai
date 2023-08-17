/** @module controller/sidebar/03-tweaks */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { Controller } from "../base.mjs";
import { 
    NumberInputView, 
    FloatInputView,
    CheckboxInputView
} from "../../view/forms/input.mjs";
import { 
    SchedulerInputView,
    MultiDiffusionSchedulerInputView
} from "../common/model-manager.mjs";

let defaultGuidanceScale = 7,
    defaultInferenceSteps = 40;

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
            },
            "scheduler": {
                "label": "Scheduler",
                "class": SchedulerInputView
            },
            "multiScheduler": {
                "label": "Multi-Diffusion Scheduler",
                "class": MultiDiffusionSchedulerInputView
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
                "inferenceSteps": defaultInferenceSteps,
                "scheduler": null,
                "multiScheduler": null
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
            this.engine.scheduler = values.scheduler;
            this.engine.multiScheduler = values.multiScheduler;
        });

        // Subscribe to model changes to look for defaults
        this.subscribe("modelPickerChange", (newModel) => {
            if (!isEmpty(newModel)) {
                let defaultConfig = newModel.defaultConfiguration,
                    tweaksConfig = {};
                
                if (!isEmpty(defaultConfig.guidance_scale)) {
                    tweaksConfig.guidanceScale = defaultConfig.guidance_scale;
                }
                if (!isEmpty(defaultConfig.inference_steps)) {
                    tweaksConfig.inferenceSteps = defaultConfig.inference_steps;
                }
                if (!isEmpty(defaultConfig.scheduler)) {
                    tweaksConfig.scheduler = defaultConfig.scheduler;
                }
                if (!isEmpty(defaultConfig.multi_scheduler)) {
                    tweaksConfig.multiScheduler = defaultConfig.multi_scheduler;
                }
                if (!isEmpty(tweaksConfig)) {
                    this.tweaksForm.setValues(tweaksConfig);
                }
            }
        });

        // Add to sidebar
        this.application.sidebar.addChild(this.tweaksForm);
    }
}

export { TweaksController as SidebarController }
