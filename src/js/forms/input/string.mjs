/** @module forms/input/strings.mjs */
import { InputView } from "./base.mjs";

/**
 * The StringInputView adds the maxlength, minlength and pattern attributes.
 */
class StringInputView extends InputView {
    /**
     * @var int The maximum length of the input.
     */
    static maxLength = 128;

    /**
     * @var string An optional regular expression that must match.
     */
    static pattern = null;

    /**
     * @var int An optional minimum length of the input.
     */
    static minLength = null;

    async build() {
        let node = await super.build();
        node.attr("maxlength", this.constructor.maxLength);
        node.attr("minlength", this.constructor.minLength);
        node.attr("pattern", this.constructor.pattern);
        return node;
    }
}

/**
 * The TextInputView renders a text area instead of a single-line input.
 */
class TextInputView extends InputView {
    /**
     * @var int The number of rows of text to display.
     */
    static rows = 5;

    /**
     * @var string Override the tag name to TextArea.
     */
    static tagName = "textarea";

    /**
     * @var int The maximum length of the input.
     */
    static maxLength = 1024;

    /**
     * Builds the textarea.
     */
    async build() {
        let node = await super.build();
        node.attr("rows", this.constructor.rows)
            .attr("maxlength", this.constructor.maxLength);

        return node;
    }
}

/**
 * Password inputs
 */
class PasswordInputView extends StringInputView {
    /**
     * @var string The input type attribute
     */
    static inputType = "password";

    /**
     * @var string The autoComplete attribute. This should prevent autofilling.
     */
    static autoComplete = "new-password";
}

export { StringInputView, TextInputView, PasswordInputView };
