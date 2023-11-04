/** @module controller/sidebar/07-prompts */
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
    getState(includeImages = true) {
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
                "usePromptTravel": false,
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
        let isAnimation = false;
        this.promptsForm = new PromptsFormView(this.config);
        this.promptsForm.onSubmit(async (values) => {
            this.engine.prompt = values.prompt;
            this.engine.negativePrompt = values.negativePrompt;
            if (values.usePromptTravel && isAnimation) {
                this.publish("promptTravelEnabled");
                this.promptsForm.addClass("use-prompt-travel");
            } else {
                this.publish("promptTravelDisabled");
                this.promptsForm.removeClass("use-prompt-travel");
            }
        });
        this.promptsForm.onShortcutSubmit(() => {
            this.application.publish("tryInvoke");
        });
        this.application.sidebar.addChild(this.promptsForm);
        this.subscribe("engineAnimationFramesChange", (frames) => {
            isAnimation = !isEmpty(frames) && frames > 0;
            if (isAnimation) {
                this.promptsForm.addClass("show-prompt-travel");
                if (this.promptsForm.values.usePromptTravel) {
                    this.publish("promptTravelEnabled");
                    this.promptsForm.addClass("use-prompt-travel");
                } else {
                    this.publish("promptTravelDisabled");
                    this.promptsForm.removeClass("use-prompt-travel");
                }
            } else {
                this.promptsForm.removeClass("show-prompt-travel");
                this.promptsForm.removeClass("use-prompt-travel");
                this.publish("promptTravelDisabled");
            }
        });
    }
}

export { PromptsController as SidebarController };
