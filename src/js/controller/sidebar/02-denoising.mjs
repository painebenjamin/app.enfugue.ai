/** @module controlletr/sidebar/03-denoising */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { DenoisingFormView } from "../../forms/enfugue/denoising.mjs";

/**
 * Extends the menu controller for state and init
 */
class DenoisingController extends Controller {
    /**
     * Get data from the generation form
     */
    getState(includeImages = true) {
        return { "denoising": this.denoisingForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "denoising": {
                "strength": 1.0
            }
        }
    }

    /**
     * Set state in the generation form
     */
    setState(newState) {
        if (!isEmpty(newState.denoising)) {
            this.denoisingForm.setValues(newState.denoising).then(() => this.denoisingForm.submit());
        }
    };

    /**
     * On init, append form
     */
    async initialize() {
        this.denoisingForm = new DenoisingFormView(this.config);
        this.denoisingForm.hide();
        this.denoisingForm.onSubmit(async (values) => {
            this.engine.strength = values.strength;
        });
        this.application.sidebar.addChild(this.denoisingForm);
        let showForDenoising = false,
            showForInpainting = false,
            checkShow = () => {
                if (showForDenoising || showForInpainting) {
                    this.denoisingForm.show();
                } else {
                    this.denoisingForm.hide();
                }
            };
        this.subscribe("layersChanged", (layers) => {
            console.log(layers);
            showForDenoising = layers.reduce((carry, item) => carry || item.denoise, false);
            checkShow();
        });
        this.subscribe("inpaintEnabled", () => { showForInpainting = true; checkShow(); });
        this.subscribe("inpaintDisabled", () => { showForInpainting = false; checkShow(); });
    }
}

export { DenoisingController as SidebarController };
