/** @module controller/sidebar/01-engine */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { Controller } from "../base.mjs";
import {
    EngineSizeInputView,
    RefinerEngineSizeInputView,
    InpainterEngineSizeInputView
} from "../common/model-manager.mjs";

/**
 * The forms that allow for engine configuration when not using preconfigured models
 */
class EngineForm extends FormView {
    /**
     * @var bool Don't show submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object The field sets for the form
     */
    static fieldSets = {
        "Engine": {
            "size": {
                "label": "Engine Size",
                "class": EngineSizeInputView,
                "config": {
                    "required": true
                }
            },
            "refinerSize": {
                "label": "Refining Engine Size",
                "class": RefinerEngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            },
            "inpainterSize": {
                "label": "Inpainting Engine Size",
                "class": InpainterEngineSizeInputView,
                "config": {
                    "required": false,
                    "value": null
                }
            }
        }
    };
}

/**
 * Extend the menu controller to bind initialize
 */
class EngineController extends Controller {
    /**
     * Return data from the engine form
     */
    getState() {
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
                "size": this.application.config.model.invocation.defaultEngineSize,
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
        this.engineForm = new EngineForm(this.config);

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
            if (isEmpty(newType) || newType === "checkpoint") {
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
            } else if (this.engine.modelType !== "checkpoint"){
                this.engineForm.hide();
            }
        });
    }
}

export { EngineController as SidebarController }
