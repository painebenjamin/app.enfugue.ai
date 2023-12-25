/** @module forms/enfugue/tweaks */
import { FormView } from "../base.mjs";
import { 
    SelectInputView,
    NumberInputView, 
    FloatInputView,
    SchedulerInputView,
    CheckboxInputView,
    SliderPreciseInputView,
    NoiseOffsetInputView,
    NoiseMethodInputView,
    BlendMethodInputView,
    BetaScheduleInputView,
} from "../input.mjs";

let defaultGuidanceScale = 6.5,
    defaultInferenceSteps = 20;

/**
 * The forms that allow for tweak inputs
 */
class TweaksFormView extends FormView {
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
                    "step": 0.01,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness."
                }
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 250,
                    "value": defaultInferenceSteps,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process but can produce better results."
                }
            },
            "scheduler": {
                "label": "Scheduler",
                "class": SchedulerInputView
            },
            "noiseOffset": {
                "label": "Noise Offset",
                "class": NoiseOffsetInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.0,
                    "tooltip": "How much to blend offset noise in with initial latents, using the chosen blend method."
                }
            },
            "noiseMethod": {
                "label": "Noise Method",
                "class": NoiseMethodInputView,
                "config": {
                    "value": "simplex"
                }
            },
            "noiseBlendMethod": {
                "label": "Blend Method",
                "class": BlendMethodInputView,
                "config": {
                    "value": "inject"
                }
            },
            "clipSkip": {
                "label": "CLIP Skip",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "max": 12,
                    "value": 0,
                    "tooltip": "This numbers controls how many layers are removed from the end of text embeddings input into the text model. Some models have been trained with a reduced number of embedding layers, and without CLIP Skip those layers will be populated by the base Stable Diffusion text encoder layers. This is of particular use for anime-style models, where including the final layers can reduce quality significantly. In general, this value should be kept at zero unless specified by the model author."
                }
            },
            "enableFreeU": {
                "label": "Enable FreeU",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "FreeU is a method to adjust the weighting of a UNet model, increasing or decreasing the attention given to various layers of the model. This does not impact memory or sampling time, and can be tweaked without needing to reload models into memory."
                }
            },
            "freeUBackbone1": {
                "label": "Backbone Weight 1",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 1.0,
                    "max": 2.0,
                    "value": 1.5,
                    "step": 0.01,
                    "tooltip": "Adjusts the weight of primary backbone features. The recommended starting value for this field is <strong>1.5</strong> for Stable Diffusion 1 or <strong>1.3</strong> for Stable Diffusion XL."
                }
            },
            "freeUBackbone2": {
                "label": "Backbone Weight 2",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 1.0,
                    "max": 2.0,
                    "value": 1.6,
                    "step": 0.01,
                    "tooltip": "Adjusts the weight of secondary backbone features. The recommended starting value for this field is <strong>1.6</strong> for Stable Diffusion 1 or <strong>1.4</strong> for Stable Diffusion XL."

                }
            },
            "freeUSkip1": {
                "label": "Skip Weight 1",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.9,
                    "step": 0.01,
                    "tooltip": "Adjusts the weight of primary skipped features. The recommended starting value for this field is <strong>0.9</strong>."
                }
            },
            "freeUSkip2": {
                "label": "Skip Weight 2",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.2,
                    "step": 0.01,
                    "tooltip": "Adjusts the weight of secondary skipped features. The recommended starting value for this field is <strong>0.2</strong>."
                }
            },
            /*
            "injectDpo": {
                "label": "Inject DPO",
                "class": CheckboxInputView,
                "config": {
                    "value": false,
                    "tooltip": "When checked, DPO (Direct Preference Optimization) weight offsets will be injected into the UNet. DPO was trained by using a large dataset of images that were selected by humans for prompt alignment and overall quality. By adding their training to your active model, it can improve overall adherence to your text."
                }
            }
            */
        }
    };

    /**
     * On submit, check if we should change CSS classes.
     */
    async submit() {
        await super.submit();
        if (this.values.enableFreeU) {
            this.addClass("show-free-u");
        } else {
            this.removeClass("show-free-u");
        }
    }

    /**
     * On set value, check if we should change CSS classes.
     */
    async setValues(values) {
        await super.setValues(values);
        if (this.values.enableFreeU) {
            this.addClass("show-free-u");
        } else {
            this.removeClass("show-free-u");
        }
    }
};

/**
 * Some advanced options for schedulers
 */
class SchedulerConfigurationFormView extends FormView {
    /**
     * @var bool Disable submit button
     */
    static autoSubmit = true;

    /**
     * @var object scheduler kwargs
     */
    static fieldSets = {
        "Beta": {
            "betaStart": {
                "class": FloatInputView,
                "label": "Beta Start",
                "config": {
                    "step": 0.00001,
                    "min": 0.0,
                    "max": 1.0,
                    "placeholder": "0.00085",
                    "tooltip": "The beginning of the beta noising schedule, roughly correlating to the final amount of noise to leave in the image when used for denoising."
                }
            },
            "betaEnd": {
                "class": FloatInputView,
                "label": "Beta End",
                "config": {
                    "tooltip": "The ending of the beta noising schedule, roughly correlating to the initial amount of noise to inject in the image when used for denoising.",
                    "placeholder": "0.012",
                    "step": 0.00001,
                    "min": 0.0,
                    "max": 1.0
                }
            },
            "betaSchedule": {
                "class": BetaScheduleInputView,
                "label": "Beta Schedule",
                "config": {
                    "tooltip": "How beta goes from beginning to end over time."
                }
            }
        }
    };
};

export { 
    TweaksFormView,
    SchedulerConfigurationFormView,
};
