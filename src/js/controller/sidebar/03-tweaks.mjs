/** @module controller/sidebar/03-tweaks */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { TweaksFormView } from "../../forms/enfugue/tweaks.mjs";

/**
 * Extend the menu controll to bind init
 */
class TweaksController extends Controller {
    /**
     * Return data from the tweaks form
     */
    getState(includeImages = true) {
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
                "guidanceScale": this.config.model.invocation.guidanceScale,
                "inferenceSteps": this.config.model.invocation.inferenceSteps,
                "scheduler": null,
            }
        }
    }

    /**
     * On initialization, append the Tweaks form
     */
    async initialize() {
        // Builds form
        this.tweaksForm = new TweaksFormView(this.config);
        this.tweaksForm.onSubmit(async (values) => {
            this.engine.guidanceScale = values.guidanceScale;
            this.engine.inferenceSteps = values.inferenceSteps;
            this.engine.scheduler = values.scheduler;
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
                if (!isEmpty(newModel.scheduler)) {
                    tweaksConfig.scheduler = newModel.scheduler[0].name;
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
