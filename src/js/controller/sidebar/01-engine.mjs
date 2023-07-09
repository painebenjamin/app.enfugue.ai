/** @module controller/sidebar/01-engine */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { Controller } from "../base.mjs";
import { 
    NumberInputView, 
    FloatInputView,
    CheckboxInputView
} from "../../view/forms/input.mjs";

let defaultEngineSize = 512;

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
     * @var object The tweak fields
     */
    static fieldSets = {
        "Engine": {
            "size": {
                "label": "Size",
                "class": NumberInputView,
                "config": {
                    "required": true,
                    "value": defaultEngineSize,
                    "min": 128,
                    "max": 2048,
                    "step": 8,
                    "tooltip": "When using chunked diffusion, this is the size of the window (in pixels) that will be encoded, decoded or inferred at once. Set the chunking size to 0 in the sidebar to disable chunked diffusion and always try to process the entire image at once."
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
                "size": defaultEngineSize
            }
        }
    };

    /**
     * On initialization, append the engine form
     */
    async initialize() {
        // Set defaults
        defaultEngineSize = this.application.config.model.invocation.defaultEngineSize;
        
        // Builds form
        this.engineForm = new EngineForm(this.config);
        this.engineForm.onSubmit(async (values) => {
            this.engine.size = values.size;
        });

        // Add to sidebar
        this.application.sidebar.addChild(this.engineForm);

        // Bind events to listen for when to show
        this.subscribe("engineModelTypeChange", (newType) => {
            if (isEmpty(newType) || newType === "checkpoint") {
                this.engineForm.show();
            } else {
                this.engineForm.hide();
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
