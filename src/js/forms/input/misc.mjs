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
     * Builds the node.
     */
    async build() {
        let node = await super.build();
        node.off("click").on("click", (e) => this.changed(e));
        return node;
    }
}

export { HiddenInputView, ButtonInputView };
