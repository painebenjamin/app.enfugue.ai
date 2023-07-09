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
 * Register controller to add to sidebar and manage state
 */
class PromptsController extends Controller {
    /**
     * When asked for state, return data from form
     */
    getState() {
        return { "prompts": this.promptsForm.values };
    }

    /**
     * Get default state
     */
    getDefaultState() {
        return { "prompts": { "prompt": null, "negativePrompt": null } };
    }

    /**
     * Set state in the prompts form
     */
    setState(newState) {
        this.promptsForm.setValues(newState.prompts).then(() => this.promptsForm.submit());
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
    }
}

export { PromptsController as SidebarController };
