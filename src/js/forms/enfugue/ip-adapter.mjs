/** @module forms/enfugue/ip-adapter */
import { FormView } from "../base.mjs";
import { SelectInputView } from "../input.mjs";

/**
 * The form class containing the strength slider
 */
class IPAdapterFormView extends FormView {
    /**
     * @var object All field sets and their config
     */
    static fieldSets = {
        "IP Adapter": {
            "ipAdapterModel": {
                "label": "Model",
                "class": SelectInputView,
                "config": {
                    "value": "default",
                    "options": {
                        "default": "Default",
                        "plus": "Plus",
                        "plus-face": "Plus Face",
                        "full-face": "Full Face",
                        "face-id": "Face ID",
                        "face-id-plus": "Face ID Plus"
                    },
                    "tooltip": "Which IP adapter model to use. 'Plus' will in general find more detail in the source image while considerably adjusting the impact of your prompt, and 'Plus Face' will ignore much of the image except for facial features. 'Full Face' is similar to 'Plus Face' but extracts more features.<br /><br />'Face ID' will extract features only from faces using the InsightFace model. 'Face ID Plus' will do this as well as the standard CLIP-based feature extraction."
                }
            }
        }
    };

    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;
};

export { IPAdapterFormView };
