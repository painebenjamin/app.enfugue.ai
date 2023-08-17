/** @module controller/sidebar/05-upscale */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { UpscaleFormView } from "../../forms/enfugue/upscale.mjs";

/**
 * The overall controller registers the form in the sidebar.
 */
class UpscaleController extends Controller {
    /**
     * When asked for state, return values from form.
     */
    getState() {
        return { "upscale": this.upscaleForm.values };
    }

    /**
     * Get default state
     */
    getDefaultState() {
        return { 
            "upscale": { 
                "outscale": 1, 
                "upscale": ["esrgan"],
                "upscaleIterative": false,
                "upscaleDiffusion": false,
                "upscaleDiffusionSteps": [100],
                "upscaleDiffusionStrength": [0.2],
                "upscaleDiffusionGuidanceScale": [12],
                "upscaleDiffusionChunkingSize": 128,
                "upscaleDiffusionChunkingBlur": 128,
                "upscaleDiffusionScaleChunkingSize": true,
                "upscaleDiffusionScaleChunkingBlur": true
            }
        };
    }

    /**
     * When setting state, look for values from the upscale form
     */
    setState(newState) {
        if (!isEmpty(newState.upscale)) {
            let upscaleState = deepClone(newState.upscale);
            if (isEmpty(upscaleState.upscaleDiffusionControlnet)) {
                upscaleState.upscaleDiffusionControlnet = [];
            }
            if (isEmpty(upscaleState.upscaleDiffusionPrompt)) {
                upscaleState.upscaleDiffusionPrompt = [];
            }
            if (isEmpty(upscaleState.upscaleDiffusionPrompt2)) {
                upscaleState.upscaleDiffusionPrompt2 = [];
            }
            if (isEmpty(upscaleState.upscaleDiffusionNegativePrompt)) {
                upscaleState.upscaleDiffusionNegativePrompt = [];
            }
            if (isEmpty(upscaleState.upscaleDiffusionNegativePrompt2)) {
                upscaleState.upscaleDiffusionNegativePrompt2 = [];
            }

            this.upscaleForm.setValues(upscaleState).then(() => this.upscaleForm.submit());
        }
    }

    /**
     * When initialized, add form to sidebar.
     */
    async initialize() {
        this.upscaleForm = new UpscaleFormView(this.config);
        this.upscaleForm.onSubmit(async (values) => {
            this.engine.upscale = values.upscale;
            this.engine.outscale = values.outscale;
            this.engine.upscaleIterative = values.upscaleIterative;
            this.engine.upscaleMethod = values.upscaleMethod;
            this.engine.upscaleDiffusion = values.upscaleDiffusion;
            this.engine.upscaleDiffusionControlnet = values.upscaleDiffusionControlnet;
            this.engine.upscaleDiffusionSteps = values.upscaleDiffusionSteps;
            this.engine.upscaleDiffusionStrength = values.upscaleDiffusionStrength;
            this.engine.upscaleDiffusionGuidanceScale = values.upscaleDiffusionGuidanceScale;
            this.engine.upscaleDiffusionChunkingSize = values.upscaleDiffusionChunkingSize;
            this.engine.upscaleDiffusionChunkingBlur = values.upscaleDiffusionChunkingBlur;
            this.engine.upscaleDiffusionScaleChunkingSize = values.upscaleDiffusionScaleChunkingSize;
            this.engine.upscaleDiffusionScaleChunkingBlur = values.upscaleDiffusionScaleChunkingBlur;
            this.engine.upscaleDiffusionPrompt = values.upscaleDiffusionPrompt;
            this.engine.upscaleDiffusionNegativePrompt = values.upscaleDiffusionNegativePrompt;
        });
        this.application.sidebar.addChild(this.upscaleForm);
    }
}

export { UpscaleController as SidebarController };
