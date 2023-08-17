/** @module forms/enfugue/tweaks */
import { FormView } from "../base.mjs";
import { 
    NumberInputView, 
    FloatInputView,
    SchedulerInputView,
    MultiDiffusionSchedulerInputView
} from "../input.mjs";

let defaultGuidanceScale = 7,
    defaultInferenceSteps = 40;

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
};

export { TweaksFormView };
