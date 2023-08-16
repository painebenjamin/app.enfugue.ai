/** @module views/nodes/image-editor/base.mjs */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../forms/base.mjs";
import { NodeView } from "../base.mjs";
import {
    TextInputView,
    FloatInputView,
    NumberInputView,
    CheckboxInputView
} from "../../forms/input.mjs";

class ImageEditorBaseOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
        "Prompts": {
            "prompt": {
                "label": "Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global prompt will be used."
                }
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global negative prompt will be used."
                }
            },
        },
        "Secondary Prompts": {
            "prompt2": {
                "label": "Secondary Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global prompt will be used. Secondary prompts are input into the secondary text encoder when using SDXL. When not using SDXL, secondary prompts will be merged with primary ones."
                }
            },
            "negativePrompt2": {
                "label": "Secondary Negative Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global negative prompt will be used. Secondary prompts are input into the secondary text encoder when using SDXL. When not using SDXL, secondary prompts will be merged with primary ones."
                }
            }
        },
        "Tweaks": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": null,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness."
                }
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 5,
                    "max": 250,
                    "step": 1,
                    "value": null,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process."
                }
            }
        },
        "Other": {
            "scaleToModelSize": {
                "label": "Scale to Model Size",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When this node has any dimension smaller than the size of the configured model, scale it up so it's smallest dimension is the same size as the model, then scale it down after diffusion.<br />This generally improves image quality in slightly rectangular shapes or square shapes smaller than the engine size, but can also result in ghosting and increased processing time.<br />This will have no effect if your node is larger than the model size in all dimensions.<br />If unchecked and your node is smaller than the model size, TensorRT will be disabled for this node."
                },
            },
            "removeBackground": {
                "label": "Remove Background",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "After diffusion, run the resulting image though an AI background removal algorithm. This can improve image consistency when using multiple nodes."
                }
            }
        },
    };
    
    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "options-form-view";

    /**
     * @var array Collapsed field sets
     */
    static collapseFieldSets = ["Secondary Prompts", "Tweaks"];
};

/**
 * Nodes on the Image Editor use multiples of 8 instead of 10
 */
class ImageEditorNodeView extends NodeView {
    /**
     * @var bool Enable header flipping
     */
    static canFlipHeader = true;

    /**
     * @var int The minimum height, much smaller than normal minimum.
     */
    static minHeight = 32;
    
    /**
     * @var int The minimum width, much smaller than normal minimum.
     */
    static minWidth = 32;

    /**
     * @var int Change snap size from 10 to 8
     */
    static snapSize = 8;

    /**
     * @var int Change padding from 10 to 8
     */
    static padding = 8;

    /**
     * @var int Change edge handler tolerance from 10 to 8
     */
    static edgeHandlerTolerance = 8;

    /**
     * @var bool All nodes on the image editor try to be as minimalist as possible.
     */
    static hideHeader = true;

    /**
     * @var string Change from 'Close' to 'Remove'
     */
    static closeText = "Remove";
    
    /**
     * @var array<object> The buttons for the node.
     * @see view/nodes/base
     */
    static nodeButtons = {
        anchor: {
            icon: "fa-solid fa-sliders",
            tooltip: "Show/Hide Options",
            callback: function() {
                this.toggleOptions();
            }
        }
    };

    /**
     * @var class The form to use. Each node should have their own.
     */
    static optionsFormView = ImageEditorBaseOptionsFormView;

    /**
     * Can be overridden in the node classes; this is called when their options are changed.
     */
    async updateOptions(values) {
        this.prompt = values.prompt;
        this.negativePrompt = values.negativePrompt;
        this.guidanceScale = values.guidanceScale;
        this.inferenceSteps = values.inferenceSteps;
        this.scaleToModelSize = values.scaleToModelSize;
        this.removeBackground = values.removeBackground;
    }

    /**
     * Shows the options view.
     */
    async toggleOptions() {
        if (isEmpty(this.optionsForm)) {
            this.optionsForm = new this.constructor.optionsFormView(this.config);
            this.optionsForm.onSubmit((values) => this.updateOptions(values));
            let optionsNode = await this.optionsForm.getNode();
            this.optionsForm.setValues(this.getState());
            this.node.find("enfugue-node-contents").append(optionsNode);
        } else if (this.optionsForm.hidden) {
            this.optionsForm.show();
        } else {
            this.optionsForm.hide();
        }
    }

    /**
     * When state is set, send to form
     */
    setState(newState) {
        super.setState({
            name: newState.name,
            x: newState.x - this.constructor.padding,
            y: newState.y - this.constructor.padding,
            h: newState.h + (this.constructor.padding * 2),
            w: newState.w + (this.constructor.padding * 2)
        });
        
        this.updateOptions(newState);
        
        if (!isEmpty(this.optionsForm)) {
            this.optionsForm.setValues(newState);
        }
    }

    /**
     * Gets the base state and appends form values.
     */
    getState() {
        let state = super.getState();
        state.prompt = this.prompt || null;
        state.negativePrompt = this.negativePrompt || null;
        state.guidanceScale = this.guidanceScale || null;
        state.inferenceSteps = this.inferenceSteps || null;
        state.removeBackground = this.removeBackground || false;
        state.scaleToModelSize = this.scaleToModelSize || false;
        return state;
    }
};

export {
    ImageEditorBaseOptionsFormView,
    ImageEditorNodeView
};
