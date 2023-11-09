/** @module forms/enfugue/prompts */
import { FormView } from "../base.mjs";
import {
    PromptInputView,
    NumberInputView,
    CheckboxInputView
} from "../input.mjs";

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
            "usePromptTravel": {
                "label": "Use Prompt Travel",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, you can change prompts throughout an animation using a timeline interface. When disabled, the same problem will be used throughout the entire animation."
                }
            },
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

/**
 * The prompt travel form is form prompts with start/end frames
 */
class PromptTravelFormView extends FormView {
    /**
     * @var bool Don't show submit button
     */
    static autoSubmit = true;

    /**
     * @var object The field sets
     */
    static fieldSets = {
        "Prompts": {
            "positive": {
                "label": "Prompt",
                "class": SubmitPromptInputView
            },
            "negative": {
                "label": "Negative Prompt",
                "class": SubmitPromptInputView
            }
        },
        "Weight": {
            "weight": {
                "class": NumberInputView,
                "config": {
                    "min": 0.01,
                    "value": 1.0,
                    "step": 0.01,
                    "tooltip": "The weight of this prompt. It is recommended to keep your highest-weight prompt at 1.0 and scale others relative to that, but this is unconstrained."
                }
            }
        }
    };
}

export {
    PromptsFormView,
    PromptTravelFormView
};
