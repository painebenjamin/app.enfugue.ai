/** @module forms/enfugue/canvas */
import { FormView } from "../base.mjs";
import { NumberInputView } from "../input.mjs";

let defaultWidth = 512,
    defaultHeight = 512,
    defaultChunkingSize = 64,
    defaultChunkingBlur = 64;

/**
 * This controls dimensions of the canvas and multidiffusion step
 */
class CanvasFormView extends FormView {
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
            "chunkingSize": {
                "label": "Chunk Size",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "value": defaultChunkingSize,
                    "step": 8,
                    "tooltip": "<p>The number of pixels to move the frame when doing chunked diffusion.</p><p>When this number is greater than 0, the engine will only ever process a square in the size of the configured model size at once. After each square, the frame will be moved by this many pixels along either the horizontal or vertical axis, and then the image is re-diffused. When this number is 0, chunking is disabled, and the entire canvas will be diffused at once.</p><p>Disabling this (setting it to 0) can have varying visual results, but a guaranteed result is drastically increased VRAM usage for large images. A low number can produce more detailed results, but can be noisy, and takes longer to process. A high number is faster to process, but can have poor results especially along frame boundaries. The recommended value is set by default.</p>"
                }
            },
            "chunkingBlur": {
                "label": "Chunk Blur",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "value": defaultChunkingBlur,
                    "step": 8,
                    "tooltip": "The number of pixels to feather along the edge of the frame when blending chunked diffusions together. Low numbers can produce less blurry but more noisy results, and can potentially result in visible breaks in the frame. High numbers can help blend frames, but produce blurrier results. The recommended value is set by default."
                }
            }
        }
    };
};

export { CanvasFormView };
