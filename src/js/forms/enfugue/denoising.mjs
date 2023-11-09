/** @module forms/enfugue/denoising */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import { SliderPreciseInputView } from "../input.mjs";

/**
 * The form class containing the strength slider
 */
class DenoisingFormView extends FormView {
    /**
     * @var object All field sets and their config
     */
    static fieldSets = {
        "Denoising Strength": {
            "strength": {
                "class": SliderPreciseInputView,
                "config": {
                    "value": 0.99,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "The amount of the image to change. A value of 1.0 means the final image will be completely different from the input image, and a value of 0.0 means the final image will not change from the input image. A value of 0.8 usually represents a good balance of changing the image while maintaining similar features."
                }
            }
        }
    };

    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;
};

export {
    DenoisingFormView
};
