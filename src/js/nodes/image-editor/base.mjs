/** @module nodes/image-editor/base.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { ImageEditorNodeOptionsFormView } from "../../forms/enfugue/image-editor.mjs";
import { NodeView } from "../base.mjs";

/**
 * Nodes on the Image Editor use multiples of 8 instead of 10
 */
class ImageEditorNodeView extends NodeView {
    /**
     * @var string The name to show in the menu
     */
    static nodeTypeName = "Base";

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
        options: {
            icon: "fa-solid fa-sliders",
            tooltip: "Show/Hide Options",
            shortcut: "o",
            callback: function() {
                this.toggleOptions();
            }
        }
    };

    /**
     * @var class The form to use. Each node should have their own.
     */
    static optionsFormView = ImageEditorNodeOptionsFormView;

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
    getState(includeImages = true) {
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
    ImageEditorNodeView
};
