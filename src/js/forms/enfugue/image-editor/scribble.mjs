/** @module forms/enfugue/image-editor/scribble */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../../forms/base.mjs";
import {
    PromptInputView,
    FloatInputView,
    NumberInputView,
    CheckboxInputView,
    SliderPreciseInputView
} from "../../../forms/input.mjs";

class ImageEditorScribbleNodeOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
        "Scribble ControlNet Parameters": {
            "conditioningScale": {
                "label": "Conditioning Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "How closely to follow the Scribble ControlNet's influence. Typical values vary, usually values between 0.5 and 1.0 produce good conditioning with balanced randomness, but other values may produce something closer to the desired result."
                }
            },
            "conditioningStart": {
                "label": "Conditioning Start",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.0,
                    "tooltip": "When to begin using the Scribble ControlNet for influence. Defaults to the beginning of generation."
                }
            },
            "conditioningEnd": {
                "label": "Conditioning End",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "When to stop using the Scribble ControlNet for influence. Defaults to the end of generation."
                }
            },
        }
    };

    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "options-form-view";
};

export { ImageEditorScribbleNodeOptionsFormView };
