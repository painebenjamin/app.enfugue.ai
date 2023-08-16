/** @module controlletr/sidebar/04-generation */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { NumberInputView, FloatInputView } from "../../view/forms/input.mjs";
import { Controller } from "../base.mjs";

/**
 * The GenerationForm controls samples and seeds
 */
class GenerationForm extends FormView {
    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object All the inputs in this controller
     */
    static fieldSets = {
        "Generation": {
            "samples": {
                "label": "Samples",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 8,
                    "value": 1,
                    "step": 1,
                    "tooltip": "The number of concurrent samples to generate. Each sample linearly increases the amount of VRAM required, but only logarithmically increases the inference time."
                }
            },
            "iterations": {
                "label": "Iterations",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "value": 1,
                    "step": 1,
                    "tooltip": "The number of times to generate samples. Each iteration will generate the passed number of samples and keep a running array of result images."
                }
            },
            "seed": {
                "label": "Seed",
                "class": NumberInputView,
                "config": {
                    "tooltip": "The initialization value for the random number generator. Set this to a number to produce consistent results with every invocation."
                }
            } // TODO: Add randomize button
        }
    };
}

/**
 * Extends the menu controller for state and init
 */
class GenerationController extends Controller {
    /**
     * Get data from the generation form
     */
    getState() {
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
        this.generationForm = new GenerationForm(this.config);
        this.generationForm.onSubmit(async (values) => {
            this.engine.samples = values.samples;
            this.engine.iterations = values.iterations;
            this.engine.seed = values.seed;
        });
        this.application.sidebar.addChild(this.generationForm);
    }
}

export { GenerationController as SidebarController };
