/** @module controller/sidebar/01-engine */
import { isEmpty } from "../../base/helpers.mjs";
import { EngineFormView } from "../../forms/enfugue/engine.mjs";
import { Controller } from "../base.mjs";

/**
 * Extend the menu controller to bind initialize
 */
class EngineController extends Controller {
    /**
     * Return data from the engine form
     */
    getState(includeImages = true) {
        return { "engine": this.engineForm.values };
    }

    /**
     * Sets state in the form
     */
    setState(newState) {
        if (!isEmpty(newState.engine)) {
            this.engineForm.setValues(newState.engine).then(() => this.engineForm.submit());
        }
    }

    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "engine": {
                "size": null,
                "refinerSize": null,
                "inpainterSize": null
            }
        }
    };

    /**
     * On initialization, append the engine form
     */
    async initialize() {
        // Builds form
        this.engineForm = new EngineFormView(this.config);

        // Bind submit
        this.engineForm.onSubmit(async (values) => {
            this.engine.size = values.size;
            this.engine.refinerSize = values.refinerSize;
            this.engine.inpainterSize = values.inpainterSize;
        });

        // Add to sidebar
        this.application.sidebar.addChild(this.engineForm);

        // Bind events to listen for when to show form and fields
        this.subscribe("engineModelTypeChange", (newType) => {
            if (isEmpty(newType) || newType !== "model") {
                this.engineForm.show();
            } else {
                this.engineForm.hide();
            }
        });
        this.subscribe("engineRefinerChange", (newRefiner) => {
            if (isEmpty(newRefiner)) {
                this.engineForm.removeClass("show-refiner");
            } else {
                this.engineForm.addClass("show-refiner");
            }
        });
        this.subscribe("engineInpainterChange", (newInpainter) => {
            if (isEmpty(newInpainter)) {
                this.engineForm.removeClass("show-inpainter");
            } else {
                this.engineForm.addClass("show-inpainter");
            }
        });
        this.subscribe("engineModelChange", (newModel) => {
            if (isEmpty(newModel)) {
                this.engineForm.show();
            } else if (this.engine.modelType === "model"){
                this.engineForm.hide();
            }
        });
    }
}

export { EngineController as SidebarController }
