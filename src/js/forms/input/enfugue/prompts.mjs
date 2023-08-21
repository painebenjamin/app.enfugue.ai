/** @module forms/input/enfugue/prompt */
import { InputView } from "../base.mjs";
import { TextInputView } from "../string.mjs";
import { ButtonInputView } from "../misc.mjs";

/**
 * Define the secondary prompt button
 */
class AddSecondaryPromptInputView extends ButtonInputView {
    /**
     * @var string The text that is displayed
     */
    static defaultValue = "Add Secondary";

    /**
     * @var string tooltip to display
     */
    static tooltip = "Secondary prompts are input into the secondary text encoder when using SDXL. When not using SDXL, secondary prompts and primary prompts will be merged.";

    /**
     * @var string css class
     */
    static className = "add-secondary-prompt"
};

/**
 * Define the remove secondary prompt button
 */
class RemoveSecondaryPromptInputView extends AddSecondaryPromptInputView {
    /**
     * @var string The text that is displayed
     */
    static defaultValue = "Remove Secondary";

    /**
     * @var string css class
     */
    static className = "remove-secondary-prompt"
};

/**
 * The PromptInputView allows for one or two prompts, 
 * either positive or negative.
 */
class PromptInputView extends InputView {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-prompt-input-view";

    /**
     * @var class Text input class
     */
    static textInputClass = TextInputView;

    /**
     * On construct, build child inputs
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.primary = new TextInputView(this.config, "primary");
        this.secondary = new TextInputView(this.config, "secondary");
        this.primary.addClass("primary-prompt");
        this.secondary.addClass("secondary-prompt");
        this.addSecondary = new AddSecondaryPromptInputView(this.config);
        this.removeSecondary = new RemoveSecondaryPromptInputView(this.config);
        this.addSecondary.onChange(() => this.addSecondaryPrompt());
        this.removeSecondary.onChange(() => this.removeSecondaryPrompt());
        this.removeSecondaryPrompt();
        this.primary.onInput(() => this.inputted());
        this.secondary.onInput(() => this.inputted());
        this.primary.onChange(() => this.changed());
        this.secondary.onChange(() => this.changed());
        this.primary.onKeyPress((e) => this.keyPressed(e));
        this.secondary.onKeyPress((e) => this.keyPressed(e));
    }

    /**
     * Gets the value, either a string or an array
     */
    getValue() {
        return this.showSecondary
            ? [this.primary.getValue(), this.secondary.getValue()]
            : this.primary.getValue();
    }

    /**
     * Sets the value, either a string or an array
     */
    setValue(newValue, triggerChange) {
        if (Array.isArray(newValue)) {
            if (newValue[1] === null) {
                this.primary.setValue(newValue[0], false);
                this.secondary.setValue("", false);
                this.removeClass("show-secondary");
            } else {
                this.addSecondaryPrompt();
                this.primary.setValue(newValue[0], false);
                this.secondary.setValue(newValue[1], false);
            }
        } else {
            this.primary.setValue(newValue, false);
        }
        super.setValue(newValue, triggerChange);
    }

    /**
     * Enables secondary prompts
     */
    addSecondaryPrompt() {
        let isChanged = this.showSecondary === true;
        this.showSecondary = true;
        this.addClass("show-secondary");
        if (isChanged) {
            this.changed();
        }
    }

    /**
     * Disables secondary prompts
     */
    removeSecondaryPrompt() {
        let isChanged = this.showSecondary === true;
        this.showSecondary = false;
        this.removeClass("show-secondary");
        if (isChanged) {
            this.changed();
        }
    }
    
    /**
     * On build, append text inputs.
     */
    async build() {
        // Set tooltip on child items to avoid conflicting tooltips
        let node = await super.build(),
            nodeTooltip = node.data("tooltip"),
            primaryInput = await this.primary.getNode(),
            secondaryInput = await this.secondary.getNode();

        node.data("tooltip", null);
        primaryInput.data("tooltip", nodeTooltip);
        secondaryInput.data("tooltip", nodeTooltip);
        return node.append(
            await this.addSecondary.getNode(),
            await this.removeSecondary.getNode(),
            primaryInput,
            secondaryInput
        );
    }
}

export { PromptInputView };
