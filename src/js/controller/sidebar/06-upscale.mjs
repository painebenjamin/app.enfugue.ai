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
    getState(includeImages = true) {
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
                "upscalePipeline": null,
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
            if (isEmpty(upscaleState.upscaleDiffusionNegativePrompt)) {
                upscaleState.upscaleDiffusionNegativePrompt = [];
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
            this.engine.upscalePipeline = values.upscalePipeline;
            this.engine.upscaleMethod = values.upscaleMethod;
            this.engine.upscaleDiffusion = values.upscaleDiffusion;
            this.engine.upscaleDiffusionPrompt = values.upscaleDiffusionPrompt;
            this.engine.upscaleDiffusionNegativePrompt = values.upscaleDiffusionNegativePrompt;
            this.engine.upscaleDiffusionControlnet = values.upscaleDiffusionControlnet;
            this.engine.upscaleDiffusionSteps = values.upscaleDiffusionSteps;
            this.engine.upscaleDiffusionStrength = values.upscaleDiffusionStrength;
            this.engine.upscaleDiffusionGuidanceScale = values.upscaleDiffusionGuidanceScale;
            this.engine.upscaleDiffusionChunkingSize = values.upscaleDiffusionChunkingSize;
            this.engine.upscaleDiffusionChunkingBlur = values.upscaleDiffusionChunkingBlur;
            this.engine.upscaleDiffusionScaleChunkingSize = values.upscaleDiffusionScaleChunkingSize;
            this.engine.upscaleDiffusionScaleChunkingBlur = values.upscaleDiffusionScaleChunkingBlur;
        });
        
        this.subscribe("modelPickerChange", (newModel) => {
            if (!isEmpty(newModel)) {
                let defaultConfig = newModel.defaultConfiguration,
                    upscaleConfig = {};

                if (!isEmpty(defaultConfig.upscale)) {
                    upscaleConfig.upscale = defaultConfig.upscale;
                }
                if (!isEmpty(defaultConfig.outscale)) {
                    upscaleConfig.outscale = defaultConfig.outscale;
                }
                if (!isEmpty(defaultConfig.upscale_iterative)) {
                    upscaleConfig.upscaleIterative = defaultConfig.upscale_iterative;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion)) {
                    upscaleConfig.upscaleDiffusion = defaultConfig.upscale_diffusion;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_steps)) {
                    upscaleConfig.upscaleDiffusionSteps = defaultConfig.upscale_diffusion_steps;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_pipeline)) {
                    upscaleConfig.upscaleDiffusionPipeline = defaultConfig.upscale_diffusion_pipeline;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_strength)) {
                    upscaleConfig.upscaleDiffusionStrength = defaultConfig.upscale_diffusion_strength;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_guidance_scale)) {
                    upscaleConfig.upscaleDiffusionGuidanceScale = defaultConfig.upscale_diffusion_guidance_scale;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_controlnet)) {
                    upscaleConfig.upscaleDiffusionControlnet = defaultConfig.upscale_diffusion_controlnet;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_chunking_size)) {
                    upscaleConfig.upscaleDiffusionChunkingSize = defaultConfig.upscale_diffusion_chunking_size;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_chunking_blur)) {
                    upscaleConfig.upscaleDiffusionChunkingBlur = defaultConfig.upscale_diffusion_chunking_blur;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_scale_chunking_size)) {
                    upscaleConfig.upscaleDiffusionScaleChunkingSize = defaultConfig.upscale_diffusion_scale_chunking_size;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_scale_chunking_blur)) {
                    upscaleConfig.upscaleDiffusionScaleChunkingBlur = defaultConfig.upscale_diffusion_scale_chunking_blur;
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_prompt)) {
                    if (!isEmpty(defaultConfig.upscale_diffusion_prompt_2)) {
                        upscaleConfig.upscaleDiffusionPrompt = [
                            defaultConfig.upscale_diffusion_prompt,
                            defaultConfig.upscale_diffusion_prompt_2
                        ];
                    } else {
                        upscaleConfig.upscaleDiffusionPrompt = defaultConfig.upscale_diffusion_prompt;
                    }
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_negative_prompt)) {
                    if (!isEmpty(defaultConfig.upscale_diffusion_negative_prompt_2)) {
                        upscaleConfig.upscaleDiffusionNegativePrompt = [
                            defaultConfig.upscale_diffusion_negative_prompt,
                            defaultConfig.upscale_diffusion_negative_prompt_2
                        ];
                    } else {
                        upscaleConfig.upscaleDiffusionNegativePrompt = defaultConfig.upscale_diffusion_negative_prompt;
                    }
                }
                
                if (!isEmpty(upscaleConfig)) {
                    this.upscaleForm.setValues(upscaleConfig);
                }
            }
        });

        this.application.sidebar.addChild(this.upscaleForm);
    }
}

export { UpscaleController as SidebarController };
