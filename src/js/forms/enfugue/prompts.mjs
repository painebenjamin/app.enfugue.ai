/** @module forms/enfugue/prompts */
import { ElementBuilder } from "../../base/builder.mjs";
import { FormView } from "../base.mjs";
import {
    PromptInputView,
    NumberInputView,
    CheckboxInputView,
    RepeatableInputView,
    TextInputView
} from "../input.mjs";

const E = new ElementBuilder();

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
        },
        "Frame Timing": {
            "start": {
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "value": 1,
                    "step": 1,
                    "tooltip": "The starting frame for this prompt."
                }
            },
            "end": {
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "value": 16,
                    "step": 1,
                    "tooltip": "The ending frame for this prompt."
                }
            }
        }
    };
}

/**
 * This small class allows repeating captions
 */
class CaptionInputView extends RepeatableInputView {
    /**
     * @var class Use text input (no secondary prompts)
     */
    static memberClass = TextInputView;

    /**
     * @var int Minimum inputs
     */
    static minimumItems = 1;
}

/**
 * The caption upsample form view lets you send one or more
 * prompts at a time to the backend for upsampling
 */
class CaptionUpsampleFormView extends FormView {
    /**
     * @var string The text to show in the form
     */
    static description = "Use this tool to transform a prompt into a more descriptive one using a large language model.<br /><br />At present, the only available model is <a href='https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha' target='_blank'>HuggingFace's 7-billion parameter Zephyr model</a>, licensed under the <a href='https://choosealicense.com/licenses/mit/' target='_blank'>MIT License.</a> This requires approximately 10Gb of hard-drive space and 12Gb of VRAM. Other open-source models will be available in the future.<br /><br />This tool follows your safety checker settings. When safety checking is disabled, this model can produce problematic outputs when prompted to do so; user discretion is advised.";

    /**
     * @var object The field sets
     */
    static fieldSets = {
        "Prompts": {
            "prompts": {
                "class": CaptionInputView
            }
        },
        "Captions Per Prompt": {
            "num_results_per_prompt": {
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "value": 1,
                    "step": 1
                 }
             }
         }
    };

    /**
     * On build, prepend text.
     */
    async build() {
        let node = await super.build();
        node.prepend(E.p().content(this.constructor.description));
        return node;
    }
}

export {
    PromptsFormView,
    PromptTravelFormView,
    CaptionUpsampleFormView
};
