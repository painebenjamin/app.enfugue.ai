/** @module forms/input/files */
import { isEmpty } from "../../base/helpers.mjs";
import { InputView } from "./base.mjs";

/**
 * This input allows for file uploads.
 */
class FileInputView extends InputView {
    /**
     * @var string The input type attribute.
     */
    static inputType = "file";

    /**
     * When a new file is put in here, resize the background so
     * the user knows the file is not uploaded.
     */
    inputted(e) {
        super.inputted(e);
        this.node.css({"background-size": "0% 100%"});
    }

    /**
     * Override getValue() to return the actual file,
     * not the filename which is what a file input element would
     * normally return.
     */
    getValue() {
        if (this.node !== undefined) {
            if (this.node.element.files && this.node.element.files.length > 0) {
                return this.node.element.files[0];
            }
        }
    }

    /**
     * Sets the upload status progress of this input.
     *
     * @param float $i The progress, between 0 and 1.
     */
    setProgress(i) {
        this.progress = i;
        if (this.node !== undefined) {
            this.node.css({
                "background-size": `${this.progress * 100}% 100%`
            });
        }
    }

    /**
     * Builds the node
     */
    async build() {
        let node = await super.build(),
            progress = isEmpty(this.progress) ? 0 : this.progress;

        return node.css({"background-size": `${progress * 100}% 100%`});
    }
}

export { FileInputView };
