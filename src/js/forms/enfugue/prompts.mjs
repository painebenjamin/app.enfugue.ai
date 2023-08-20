/** @module forms/enfugue/prompts */
import { FormView } from "../base.mjs";
import { PromptInputView } from "../input.mjs";

/**
 * Extends the prompt input view to look for ctrl+enter to auto-submit parent form
 */
class SubmitPromptInputView extends PromptInputView {
    /**
     * On key press, look for ctrl+enter
     */
    async keyPressed(e) {
        await super.keyPressed(e);
        if (e.code === "Enter" && e.ctrlKey) {
            this.form.shortcutSubmit();
        }
    }
};

/**
 * The prompts form is always shown and allows for two text inputs
 */
class PromptsFormView extends FormView {
    /**
     * Create an array for callbacks
     */
    constructor(config, values) {
        super(config, values);
        this.shortcutSubmitCallbacks = [];
    }

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
                "class": SubmitPromptInputView
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": SubmitPromptInputView
            }
        }
    };

    /**
     * Allows adding callbacks for shortcut submit
     */
    onShortcutSubmit(callback){
        this.shortcutSubmitCallbacks.push(callback);
    }

    /**
     * On shortcut submit, trigger submit
     */
    async shortcutSubmit() {
        await this.submit();
        for (let callback of this.shortcutSubmitCallbacks) {
            await callback();
        }
    }
}

export { PromptsFormView };
