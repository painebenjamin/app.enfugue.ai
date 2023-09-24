/** @module forms/enfugue/canvas */
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    CheckboxInputView,
    MaskTypeInputView
} from "../input.mjs";

let defaultWidth = 512,
    defaultHeight = 512,
    defaultChunkingSize = 64,
    defaultChunkingBlur = 64;

/**
 * This controls dimensions of the canvas and multidiffusion step
 */
class CanvasFormView extends FormView {
    /**
     * @var string Custom CSS Class
     */
    static className = "canvas-form-view";

    /**
     * @var bool Collapse these fields by default
     */
    static collapseFieldSets = true;

    /**
     * @var bool Hide submit button
     */
    static autoSubmit = true;

    /**
     * @var object Define the three field inputs
     */
    static fieldSets = {
        "Dimensions": {
            "width": {
                "label": "Width",
                "class": NumberInputView,
                "config": {
                    "min": 64,
                    "max": 16384,
                    "value": defaultWidth,
                    "step": 8,
                    "tooltip": "The width of the canvas in pixels."
                }
            },
            "height": {
                "label": "Height",
                "class": NumberInputView,
                "config": {
                    "min": 64,
                    "max": 16384,
                    "value": defaultHeight,
                    "step": 8,
                    "tooltip": "The height of the canvas in pixels."
                }
            },
            "tileHorizontally": {
                "label": "Horizontally Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile horizontally, i.e., when duplicated and placed side-by-side, there will be no seams between the copies."
                }
            },
            "tileVertically": {
                "label": "Vertically Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile vertically, i.e., when duplicated and placed with on image on top of the other, there will be no seams between the copies."
                }
            },
            "useChunking": {
                "label": "Use Chunking",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the engine will only ever process a square in the size of the configured model size at once. After each square, the frame will be moved by the configured amount of pixels along either the horizontal or vertical axis, and then the image is re-diffused. When this is disabled, the entire canvas will be diffused at once. This can have varying results, but a guaranteed result is increased VRAM use.",
                    "value": true
                }
            },
            "chunkingSize": {
                "label": "Chunking Size",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "value": defaultChunkingSize,
                    "step": 8,
                    "tooltip": "The number of pixels to move the frame when doing chunked diffusion. A low number can produce more detailed results, but can be noisy, and takes longer to process. A high number is faster to process, but can have poor results especially along frame boundaries. The recommended value is set by default."
                }
            },
            "chunkingMaskType": {
                "label": "Chunking Mask",
                "class": MaskTypeInputView
            }
        }
    };

    /**
     * On submit, add/remove CSS class for hiding/showing
     */
    async submit() {
        await super.submit();
        let chunkInput = (await this.getInputView("useChunking"));
        if (this.values.tileHorizontally || this.values.tileVertically) {
            this.removeClass("no-chunking");
            chunkInput.setValue(true, false);
            chunkInput.disable();
        } else {
            chunkInput.enable();
            if (this.values.useChunking) {
                this.removeClass("no-chunking");
            } else {
                this.addClass("no-chunking");
            }
        }
    }
};

export { CanvasFormView };
