/** @module controller/sidebar/01-canvas */
import { isEmpty } from "../../base/helpers.mjs";
import { CanvasFormView } from "../../forms/enfugue/canvas.mjs";
import { Controller } from "../base.mjs";

/**
 * Extend the Controller to put the form in the sidebar and trigger changes.
 */
class CanvasController extends Controller {
    /**
     * Get state from the form
     */
    getState(includeImages = true) {
        return { "canvas": this.canvasForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "canvas": {
                "tileHorizontal": false,
                "tileVertical": false,
                "width": this.config.model.invocation.width,
                "height": this.config.model.invocation.height,
                "tilingUnet": false,
                "tilingVae": false,
                "tilingSize": this.config.model.invocation.tilingSize,
                "tilingStride": this.config.model.invocation.tilingStride,
                "tilingMaskType": this.config.model.invocation.tilingMaskType
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
        // Create form
        this.canvasForm = new CanvasFormView(this.config);
        this.canvasForm.onSubmit(async (values) => {
            if (!this.images.hasClass("has-sample")) {
                this.images.setDimension(values.width, values.height);
            }
            this.engine.width = values.width;
            this.engine.height = values.height;
            this.engine.tileHorizontal = values.tileHorizontal;
            this.engine.tileVertical = values.tileVertical;
            this.engine.tilingUnet = values.tilingUnet;
            this.engine.tilingVae = values.tilingVae;
            if (values.tilingUnet || values.tilingVae || values.tileHorizontal || values.tileVertical) {
                this.engine.tilingSize = values.tilingSize;
                this.engine.tilingMaskType = values.tilingMaskType;
                this.engine.tilingStride = isEmpty(values.tilingStride) ? 64 : values.tilingStride;
            } else {
                this.engine.tilingStride = 0;
            }
        });

        // Add form to sidebar
        this.application.sidebar.addChild(this.canvasForm);

        // Add a callback when the image dimension is manually set
        this.images.onSetDimension((newWidth, newHeight) => {
            let currentState = this.canvasForm.values;
            currentState.width = newWidth;
            currentState.height = newHeight;
            this.canvasForm.setValues(currentState).then(() => this.canvasForm.submit());
        });

        // Trigger once the app is ready to change shapes as needed
        this.subscribe("applicationReady", () => this.canvasForm.submit());
    }
}

export { CanvasController as SidebarController };
