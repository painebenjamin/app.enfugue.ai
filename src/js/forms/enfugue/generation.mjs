/** @module forms/enfugue/generation */
import { FormView } from "../base.mjs";
import { NumberInputView } from "../input.mjs";

/**
 * The GenerationFormView controls samples and seeds
 */
class GenerationFormView extends FormView {
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

export { GenerationFormView };
