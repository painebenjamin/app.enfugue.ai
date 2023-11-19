/** @module controlletr/sidebar/09-detailing */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { DetailingFormView } from "../../forms/enfugue/detailing.mjs";

/**
 * Extends the menu controller for state and init
 */
class DetailingController extends Controller {
    /**
     * Get data from the detailing form
     */
    getState(includeImages = true) {
        return { "detailing": this.detailingForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "detailing": {
                "faceRestore": false,
                "faceInpaint": false,
                "handInpaint": false,
                "inpaintStrength": 0.25,
                "detailStrength": 0.0,
                "detailInferenceSteps": null,
                "detailGuidanceScale": null,
                "detailControlnet": null,
                "detailControlnetScale": null
            }
        };
    }

    /**
     * Set state in the detailing form
     */
    setState(newState) {
        if (!isEmpty(newState.detailing)) {
            this.detailingForm.setValues(newState.detailing).then(
                () => this.detailingForm.submit()
            );
        }
    };

    /**
     * On init, append form and hide until SDXL gets selected
     */
    async initialize() {
        this.detailingForm = new DetailingFormView(this.config);
        this.detailingForm.onSubmit(async (values) => {
            this.engine.detailerFaceRestore = values.faceRestore;
            this.engine.detailerFaceInpaint = values.faceInpaint;
            this.engine.detailerHandInpaint = values.handInpaint;
            this.engine.detailerInpaintStrength = values.inpaintStrength;
            this.engine.detailerStrength = values.detailStrength;
            this.engine.detailerGuidanceScale = values.detailGuidanceScale;
            this.engine.detailerInferenceSteps = values.detailInferenceSteps;
            this.engine.detailerControlnet = values.detailControlnet;
            this.engine.detailerControlnetScale = values.detailControlnetScale;
        });
        this.application.sidebar.addChild(this.detailingForm);
    }
}

export { DetailingController as SidebarController };
