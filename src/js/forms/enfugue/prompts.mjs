/** @module forms/enfugue/prompts */
import { FormView } from "../base.mjs";
import { PromptInputView } from "../input.mjs";

/**
 * The prompts form is always shown and allows for two text inputs
 */
class PromptsFormView extends FormView {
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
                "class": PromptInputView
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": PromptInputView
            }
        }
    };
}

export { PromptsFormView };
