/** @module controller/sidebar/06-prompts */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { TextInputView } from "../../view/forms/input.mjs";
import { Controller } from "../base.mjs";

/**
 * The prompts form is always shown and allows for two text inputs
 */
class PromptsForm extends FormView {
    /**
     * @var bool Don't show submit button
     */
    static autoSubmit = true;

    /**
     * @var object The field sets
     */
    static fieldSets = {
        "Prompts": {
            "prompt": {
                "label": "Prompt",
                "class": TextInputView
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": TextInputView
            }
        }
    };
}

/**
 * The secondary prompts form is hidden and allows for two text inputs
 */
class SecondaryPromptsForm extends FormView {
    /**
     * @var bool Don't show submit button
     */
    static autoSubmit = true;

    /**
     * @var bool Hide these fields
     */
    static collapseFieldSets = true;

    /**
     * @var object The field sets
     */
    static fieldSets = {
        "Secondary Prompts": {
            "prompt2": {
                "label": "Secondary Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "Secondary prompts are used with the secondary text encoder present in SDXL. Using two separate prompts for the two text encoders can allow for additional control over image generation.<br />When you are not using SDXL, this prompt will be merged with the main prompt."
                }
            },
            "negativePrompt2": {
                "label": "Secondary Negative Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "Secondary negative prompts are used with the secondary text encoder present in SDXL. Using two separate negative prompts for the two text encoders can allow for additional control over image generation.<br />When you are not using SDXL, this negative prompt will be merged with the main negative prompt."
                }
            }
        }
    };
}

/**
 * Register controller to add to sidebar and manage state
 */
class PromptsController extends Controller {
    /**
     * When asked for state, return data from form
     */
    getState() {
        return { 
            "prompts": {
                ...this.promptsForm.values,
                ...this.secondaryPromptsForm.values
            }
        };
    }

    /**
     * Get default state
     */
    getDefaultState() {
        return {
            "prompts": {
                "prompt": null,
                "negativePrompt": null,
                "prompt2": null,
                "negativePrompt2": null
            }
        };
    }

    /**
     * Set state in the prompts form
     */
    setState(newState) {
        Promise.all([
            this.promptsForm.setValues(newState.prompts),
            this.secondaryPromptsForm.setValues(newState.prompts)
        ]).then(() => {
            this.promptsForm.submit();
            this.secondaryPromptsForm.submit();
        });
    }

    /**
     * On init, append fields
     */
    async initialize() {
        this.promptsForm = new PromptsForm(this.config);
        this.promptsForm.onSubmit(async (values) => {
            this.engine.prompt = values.prompt;
            this.engine.negativePrompt = values.negativePrompt;
        });
        this.application.sidebar.addChild(this.promptsForm);
        
        this.secondaryPromptsForm = new SecondaryPromptsForm(this.config);
        this.secondaryPromptsForm.onSubmit(async (values) => {
            this.engine.prompt2 = values.prompt2;
            this.engine.negativePrompt2 = values.negativePrompt2;
        });
        this.application.sidebar.addChild(this.secondaryPromptsForm);
    }
}

export { PromptsController as SidebarController };
