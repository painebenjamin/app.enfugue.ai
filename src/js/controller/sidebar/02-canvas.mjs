/** @module controller/sidebar/02-canvas */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { NumberInputView } from "../../view/forms/input.mjs";
import { Controller } from "../base.mjs";

let defaultWidth = 512,
    defaultHeight = 512,
    defaultChunkingSize = 64,
    defaultChunkingBlur = 64;

/**
 * This controls dimensions of the canvas and multidiffusion step
 */
class CanvasForm extends FormView {
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
                    "max": 4096,
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
                    "max": 4096,
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
}

/**
 * Extend the Controller to put the form in the sidebar and trigger changes.
 */
class CanvasController extends Controller {
    /**
     * Get state from the form
     */
    getState() {
        return { "canvas": this.canvasForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "canvas": {
                "width": defaultWidth,
                "height": defaultHeight,
                "chunkingSize": defaultChunkingSize,
                "chunkingBlur": defaultChunkingBlur
            }
        };
    }

    /**
     * Set state on the form
     */
    setState(newState) {
        if (!isEmpty(newState.canvas)) {
            this.canvasForm.setValues(newState.canvas).then(() => this.canvasForm.submit());
        }
    }

    /**
     * On initialize, create form and bind events.
     */
    async initialize() {
        // Set defaults
        defaultWidth = this.application.config.model.invocation.width;
        defaultHeight = this.application.config.model.invocation.height;
        defaultChunkingSize = this.application.config.model.invocation.chunkingSize;
        defaultChunkingBlur = this.application.config.model.invocation.chunkingBlur;

        // Create form
        this.canvasForm = new CanvasForm(this.config);
        this.canvasForm.onSubmit(async (values) => {
            this.images.width = values.width;
            this.images.height = values.height;
            this.engine.width = values.width;
            this.engine.height = values.height;
            this.engine.chunkingSize = values.chunkingSize
            this.engine.chunkingBlur = values.chunkingBlur;
        });

        // Add form to sidebar
        this.application.sidebar.addChild(this.canvasForm);
        
        // Subscribe to model changes to look for defaults
        this.subscribe("modelPickerChange", (newModel) => {
            if (!isEmpty(newModel)) {
                let defaultConfig = newModel.defaultConfiguration,
                    canvasConfig = {};
                
                if (!isEmpty(defaultConfig.width)) {
                    canvasConfig.width = defaultConfig.width;
                }
                if (!isEmpty(defaultConfig.height)) {
                    canvasConfig.height = defaultConfig.height;
                }
                if (!isEmpty(defaultConfig.chunking_size)) {
                    canvasConfig.chunkingSize = defaultConfig.chunking_size;
                }
                if (!isEmpty(defaultConfig.chunking_blur)) {
                    canvasConfig.chunkingBlur = defaultConfig.chunking_blur;
                }
                if (!isEmpty(canvasConfig)) {
                    this.canvasForm.setValues(canvasConfig);
                }
            }
        });
    }
}

export { CanvasController as SidebarController };
