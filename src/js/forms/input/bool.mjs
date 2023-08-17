/** @module forms/input/bool */
import { InputView } from "./base.mjs";

/**
 * Extends the InputView to render a checkbox.
 */
class CheckboxInputView extends InputView {
    /**
     * @var string The input type attribute
     */
    static inputType = "checkbox";

    /**
     * Sets the value of the checkbox to true or false.
     *
     * @param bool $newValue The new value to set.
     * @param bool $triggerChange Whether or not to trigger change events.
     */
    setValue(newValue, triggerChange) {
        if (typeof newValue !== "boolean") {
            if (typeof newValue === "string") {
                newValue = ["true", "t", "yes", "y"].indexOf(newValue.toLowerCase()) !== -1;
            } else {
                newValue = !!newValue;
            }
        }
        return super.setValue(newValue, triggerChange);
    }

    /**
     * @return bool Whether or not this is checked.
     */
    getValue() {
        if (this.node !== undefined && this.node.element !== undefined) {
            return this.node.element.checked;
        }
        return !!this.value;
    }

    /**
     * Toggles the value between T/F
     */
    toggleValue() {
        if (!this.disabled) {
            this.setValue(!this.getValue());
        }
    }

    /**
     * Override labelClicked() to toggle instead of focus
     */
    labelClicked() {
        this.toggleValue();
    }
};

export { CheckboxInputView };
