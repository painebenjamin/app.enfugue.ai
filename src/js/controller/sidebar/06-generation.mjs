/** @module controlletr/sidebar/03-generation */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { GenerationFormView } from "../../forms/enfugue/generation.mjs";

/**
 * Extends the menu controller for state and init
 */
class GenerationController extends Controller {
    /**
     * Get data from the generation form
     */
    getState(includeImages = true) {
        return { "generation": this.generationForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "generation": {
                "samples": 1,
                "iterations": 1,
                "seed": null
            }
        }
    }

    /**
     * Set state in the generation form
     */
    setState(newState) {
        if (!isEmpty(newState.generation)) {
            this.generationForm.setValues(newState.generation).then(() => this.generationForm.submit());
        }
    };

    /**
     * On init, append form
     */
    async initialize() {
        this.generationForm = new GenerationFormView(this.config);
        this.generationForm.onSubmit(async (values) => {
            this.engine.samples = values.samples;
            this.engine.iterations = values.iterations;
            this.engine.seed = values.seed;
        });
        this.application.sidebar.addChild(this.generationForm);
    }
}

export { GenerationController as SidebarController };
