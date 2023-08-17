/** @module forms/input/misc */
import { InputView } from "./base.mjs";

/**
 * This small extension of the InputView simply sets type to "hidden".
 */
class HiddenInputView extends InputView {
    /**
     * @var string The input type
     */
    static inputType = "hidden";
}

/**
 * This small extension of the InputView sets the type to "button" and
 * shortcuts getting the value, since the value of a button can never change.
 */
class ButtonInputView extends InputView {
    /**
     * @var string The input type
     */
    static inputType = "button";

    /**
     * @return mixed Shortcut getValue to always return the memory value.
     */
    getValue() {
        return this.value;
    }

    /**
     * @param mixed $newValue The new value to set. Will not be set to the DOM.
     * @param bool $triggerChange Whether or not to trigger change; default false.
     */
    setValue(newValue, triggerChange) {
        this.value = newValue;
        if (triggerChange) {
            this.changed();
        }
    }

    /**
     * Builds the node.
     */
    async build() {
        let valueBeforeBuild = this.value;
        this.value = this.constructor.defaultValue;

        let node = await super.build();
        this.value = valueBeforeBuild;

        node.off("click").on("click", (e) => this.changed(e));
        return node;
    }
}

export { HiddenInputView, ButtonInputView };
