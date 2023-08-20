/** @module controller/sidebar/06-prompts */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { PromptsFormView } from "../../forms/enfugue/prompts.mjs";

/**
 * Register controller to add to sidebar and manage state
 */
class PromptsController extends Controller {
    /**
     * When asked for state, return data from form
     */
    getState() {
        return { 
            "prompts": this.promptsForm.values
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
            }
        };
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
        this.promptsForm = new PromptsFormView(this.config);
        this.promptsForm.onSubmit(async (values) => {
            this.engine.prompt = values.prompt;
            this.engine.negativePrompt = values.negativePrompt;
        });
        this.promptsForm.onShortcutSubmit(() => {
            this.application.publish("tryInvoke");
        });
        this.application.sidebar.addChild(this.promptsForm);
    }
}

export { PromptsController as SidebarController };
