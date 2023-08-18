/** @module controlletr/sidebar/05-refining */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { RefiningFormView } from "../../forms/enfugue/refining.mjs";

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
                "refinerNegativeAestheticScore": 2.5,
                "refinerPrompt": null,
                "refinerNegativePrompt": null
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
        this.refiningForm = new RefiningFormView(this.config);
        this.refiningForm.onSubmit(async (values) => {
            this.engine.refinerStrength = values.refinerStrength;
            this.engine.refinerGuidanceScale = values.refinerGuidanceScale;
            this.engine.refinerAestheticScore = values.refinerAestheticScore;
            this.engine.refinerNegativeAestheticScore = values.refinerNegativeAestheticScore;
            this.engine.refinerPrompt = values.refinerPrompt;
            this.engine.refinerNegativePrompt = values.refinerNegativePrompt;
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
