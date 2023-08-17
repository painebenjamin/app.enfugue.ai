/** @module form/confirm */
import { FormView } from "./base.mjs";
import { ElementBuilder } from "../base/builder.mjs";

const E = new ElementBuilder();

/**
 * The ConfirmFormView is a simple Form that can be submitted with no values.
 */
class ConfirmFormView extends FormView {
    /**
     * @var bool Show the cancel button.
     */
    static showCancel = true;

    /**
     * @param object $config The base configuration
     * @param string $message The message to display
     */
    constructor(config, message) {
        super(config);
        this.message = message;
    }

    /**
     * On build, display message.
     */
    async build() {
        let node = await super.build();
        node.prepend(E.p().class("confirm").content(this.message));
        return node;
    }
};

/**
 * The YesNoFormView is similar to the confirm form view but uses a yes/no syntax
 */
class YesNoFormView extends ConfirmFormView {
    /**
     * @var string Replace submit with 'yes'
     */
    static submitLabel = "Yes";
    
    /**
     * @var string Replace cancel with 'no'
     */
    static cancelLabel = "No";

    /**
     * Alias for onSubmit
     */
    onYes(callback) {
        this.onSubmit(callback);
    }

    /**
     * Alias for onCancel
     */
    onNo(callback) {
        this.onCancel(callback);
    }
};

export { ConfirmFormView, YesNoFormView };
