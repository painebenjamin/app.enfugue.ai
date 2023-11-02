/** @module forms/enfugue/canvas */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    CheckboxInputView,
    SelectInputView,
    MaskTypeInputView,
    EngineSizeInputView,
} from "../input.mjs";

let defaultWidth = 512,
    defaultHeight = 512,
    defaultChunkingSize = 64;

if (
    !isEmpty(window.enfugue) &&
    !isEmpty(window.enfugue.config) &&
    !isEmpty(window.enfugue.config.model) &&
    !isEmpty(window.enfugue.config.model.invocation)
) {
    let invocationConfig = window.enfugue.config.model.invocation;
    if (!isEmpty(invocationConfig.width)) {
        defaultWidth = invocationConfig.width;
    }
    if (!isEmpty(invocationConfig.height)) {
        defaultHeight = invocationConfig.height;
    }
    if (!isEmpty(invocationConfig.chunkingSize)) {
        defaultChunkingSize = invocationConfig.chunkingSize;
    }
}

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
                    "tooltip": "The width of the canvas in pixels.",
                    "allowNull": false
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
                    "tooltip": "The height of the canvas in pixels.",
                    "allowNull": false
                }
            },
            "tileHorizontal": {
                "label": "Horizontally Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile horizontally, i.e., when duplicated and placed side-by-side, there will be no seams between the copies."
                }
            },
            "tileVertical": {
                "label": "Vertically Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile vertically, i.e., when duplicated and placed with on image on top of the other, there will be no seams between the copies."
                }
            },
            "useChunking": {
                "label": "Enabled Tiled Diffusion/VAE",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the engine will only ever process a square in the size of the configured model size at once. After each square, the frame will be moved by the configured amount of pixels along either the horizontal or vertical axis, and then the image is re-diffused. When this is disabled, the entire canvas will be diffused at once. This can have varying results, but a guaranteed result is increased VRAM use.",
                    "value": false
                }
            },
            "size": {
                "label": "Tile Size",
                "class": EngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            },
            "chunkingSize": {
                "label": "Tile Stride",
                "class": SelectInputView,
                "config": {
                    "options": ["8", "16", "32", "64", "128", "256", "512"],
                    "value": `${defaultChunkingSize}`,
                    "tooltip": "The number of pixels to move the frame when doing tiled diffusion. A low number can produce more detailed results, but can be noisy, and takes longer to process. A high number is faster to process, but can have poor results especially along frame boundaries. The recommended value is set by default."
                }
            },
            "chunkingMaskType": {
                "label": "Tile Mask",
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
        if (this.values.tileHorizontal || this.values.tileVertical) {
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
