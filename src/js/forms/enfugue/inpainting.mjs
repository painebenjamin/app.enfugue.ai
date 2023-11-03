/** @module forms/enfugue/prompts */
import { FormView } from "../base.mjs";
import {
    CheckboxInputView,
    NumberInputView
} from "../input.mjs";

/**
 * The prompts form is always shown and allows for two text inputs
 */
class InpaintingFormView extends FormView {
    /**
     * @var bool Don't show submit button
     */
    static autoSubmit = true;

    /**
     * @var object The field sets
     */
    static fieldSets = {
        "Inpainting": {
            "outpaint": {
                "label": "Enable Outpainting",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "When enabled, enfugue will automatically fill any transparency remaining after merging layers down. If there is no transparency, this step will be skipped."
                }
            },
            "inpaint": {
                "label": "Enable Inpainting",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, you can additionally draw your own mask over any images on the canvas. If there is transparency, it will also be filled in addition to anywhere you draw."
                }
            },
            "cropInpaint": {
                "label": "Use Cropped Inpainting/Outpainting",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, enfugue will crop to the inpainted region before executing. This saves processing time and can help with small modifications on large images.",
                    "value": true
                }
            },
            "inpaintFeather": {
                "label": "Feather Amount",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "max": 512,
                    "value": 32,
                    "step": 1,
                    "tooltip": "The number of pixels to use as a blending region for cropped inpainting. These will be blended smoothly into the final image to relieve situations where the cropped inpaint is noticably different from the rest of the image."
                }
            }
        }
    };

    /**
     * Check classes on submit
     */
    async submit() {
        await super.submit();

        if (this.values.inpaint || this.values.outpaint) {
            this.removeClass("no-inpaint");
        } else {
            this.addClass("no-inpaint");
        }

        if (this.values.cropInpaint) {
            this.removeClass("no-cropped-inpaint");
        } else {
            this.addClass("no-cropped-inpaint");
        }

        let outpaintInput = (await this.getInputView("outpaint"));
        if (this.values.inpaint) {
            outpaintInput.setValue(true, false);
            outpaintInput.disable();
        } else {
            outpaintInput.enable();
        }
    }
}

export { InpaintingFormView };
