/** @module forms/enfugue/engine */
import { FormView } from "../base.mjs";
import {
    EngineSizeInputView,
    RefinerEngineSizeInputView,
    InpainterEngineSizeInputView
} from "../input.mjs";

/**
 * The forms that allow for engine configuration when not using preconfigured models
 */
class EngineFormView extends FormView {
    /**
     * @var bool Don't show submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object The field sets for the form
     */
    static fieldSets = {
        "Engine": {
            "size": {
                "label": "Engine Size",
                "class": EngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            },
            "refinerSize": {
                "label": "Refining Engine Size",
                "class": RefinerEngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            },
            "inpainterSize": {
                "label": "Inpainting Engine Size",
                "class": InpainterEngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            }
        }
    };
};

export { EngineFormView };
