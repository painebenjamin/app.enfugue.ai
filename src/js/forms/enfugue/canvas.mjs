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
    defaultTilingStride = 128;

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
    if (!isEmpty(invocationConfig.tilingSize)) {
        defaultTilingStride = invocationConfig.tilingSize;
    }
}

/**
 * Canvas size options
 */
class CanvasSizeInputView extends SelectInputView {
    /**
     * @var object Dimension options
     */
    static defaultOptions = {
        "512_512": "512×512",
        "512_768": "512×768",
        "768_512": "768×512",
        "768_768": "768×768",
        "768_1024": "768×1024",
        "768_1344": "768×1344",
        "896_1152": "896×1152",
        "1024_768": "1024×768",
        "1024_1024": "1024×1024",
        "1152_896": "1152×896",
        "1216_832": "1216×832",
        "1280_720": "1280×720",
        "1344_768": "1344×768",
        "1440_1080": "1440×1080",
        "1536_640": "1536×640",
        "1920_1080": "1920×1080",
        "2048_1080": "2048×1080",
        "3840_2160": "3840×2160",
        "4096_2160": "4096×2160",
        "custom": "Custom"
    };
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
     * @var bool Hide submit button
     */
    static autoSubmit = true;

    /**
     * @var object Define the three field inputs
     */
    static fieldSets = {
        "Dimensions": {
            "size": {
                "label": "Size",
                "class": CanvasSizeInputView,
                "config": {
                    "value": "512_512"
                }
            },
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
                "label": "Horizontally<br/>Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile horizontally, i.e., when duplicated and placed side-by-side, there will be no seams between the copies."
                }
            },
            "tileVertical": {
                "label": "Vertically<br/>Tiling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the resulting image will tile vertically, i.e., when duplicated and placed with on image on top of the other, there will be no seams between the copies."
                }
            },
            "useTiling": {
                "label": "Enabled Tiled Diffusion/VAE",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the engine will only ever process a square in the size of the configured model size at once. After each square, the frame will be moved by the configured amount of pixels along either the horizontal or vertical axis, and then the image is re-diffused. When this is disabled, the entire canvas will be diffused at once. This can have varying results, but a guaranteed result is increased VRAM use.",
                    "value": false
                }
            },
            "tilingSize": {
                "label": "Tile Size",
                "class": EngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            },
            "tilingStride": {
                "label": "Tile Stride",
                "class": SelectInputView,
                "config": {
                    "options": ["8", "16", "32", "64", "128", "256", "512"],
                    "value": `${defaultTilingStride}`,
                    "tooltip": "The number of pixels to move the frame when doing tiled diffusion. A low number can produce more detailed results, but can be noisy, and takes longer to process. A high number is faster to process, but can have poor results especially along frame boundaries. The recommended value is set by default."
                }
            },
            "tilingMaskType": {
                "label": "Tile Mask",
                "class": MaskTypeInputView
            }
        }
    };

    /**
     * Intercept inputChanged to see if it was size
     */
    async inputChanged(fieldName, fieldInput) {
        if (fieldName === "size") {
            let value = fieldInput.getValue();
            if (value !== "custom") {
                let [width, height] = value.split("_"),
                    widthInputView = await this.getInputView("width"),
                    heightInputView = await this.getInputView("height");
                width = parseInt(width);
                height = parseInt(height);
                widthInputView.setValue(width, false);
                heightInputView.setValue(height, false);
                this.values.width = width;
                this.values.height = height;
            }
        }
        return await super.inputChanged(fieldName, fieldInput);
    }

    /**
     * On submit, add/remove CSS class for hiding/showing
     */
    async submit() {
        await super.submit();

        let chunkInput = await this.getInputView("useTiling"),
            widthInput = await this.getInputView("width"),
            heightInput = await this.getInputView("height");

        if (this.values.tileHorizontal || this.values.tileVertical) {
            this.removeClass("no-tiling");
            chunkInput.setValue(true, false);
            chunkInput.disable();
        } else {
            chunkInput.enable();
            if (this.values.useTiling) {
                this.removeClass("no-tiling");
            } else {
                this.addClass("no-tiling");
            }
        }

        if (
            isEmpty(this.values.size) &&
            !isEmpty(this.values.width) &&
            !isEmpty(this.values.height)
        ) {
            let sizeInput = await this.getInputView("size"),
                size = `${this.values.width}_${this.values.height}`;
            if (isEmpty(CanvasSizeInputView.defaultOptions[size])) {
                sizeInput.setValue("custom", false);
                this.values.size = "custom";
                this.addClass("custom-size");
            } else {
                sizeInput.setValue(size, false);
                this.values.size = size;
                this.removeClass("custom-size");
            }
        } else if (this.values.size === "custom") {
            this.addClass("custom-size");
        } else if (!isEmpty(this.values.size)) {
            this.removeClass("custom-size");
            let [width, height] = this.values.size.split("_");
            widthInput.setValue(parseInt(width), false);
            heightInput.setValue(parseInt(height), false);
            this.values.width = width;
            this.values.height = height;
        }
    }

    /**
     * On set value, add/remove css classes and check preconfigured size
     */
    async setValues(values) {
        if (!isEmpty(values.width) && !isEmpty(values.height)) {
            let size = `${values.width}_${values.height}`;
            if (isEmpty(CanvasSizeInputView.defaultOptions[size])) {
                values.size = "custom";
                this.addClass("custom-size");
            } else {
                values.size = size;
                this.removeClass("custom-size");
            }
        } else if(!isEmpty(values.size)) {
            if (values.size !== "custom") {
                let [width, height] = values.size.split("_");
                values.width = parseInt(width);
                values.height = parseInt(height);
                this.removeClass("custom-size");
            } else {
                this.addClass("custom-size");
            }
        }
        return await super.setValues(values);
    }
};

export { CanvasFormView };
