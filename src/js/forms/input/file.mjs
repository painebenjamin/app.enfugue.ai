/** @module forms/input/files */
import { isEmpty } from "../../base/helpers.mjs";
import { ImageView } from "../../view/image.mjs";
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
                "background-size": `${this.progress*100}% 100%`
            });
        }
    }

    /**
     * Builds the node
     */
    async build() {
        let node = await super.build(),
            progress = isEmpty(this.progress) ? 0 : this.progress;

        node.css({"background-size": `${progress*100}% 100%`})
            .on("drop", (e) => {
                if (e.dataTransfer.files) {
                    e.stopPropagation();
                }
            });
        return node;
    }
}

/**
 * This class extends the file input view to be specific to images.
 * It also shows the image in the UI.
 */
class ImageFileInputView extends FileInputView {
    /**
     * Override getValue() to return the data if read
     */
    getValue() {
        if (!isEmpty(this.data)) {
            return this.data;
        }
        return super.getValue();
    }

    /**
     * Extend .changed() to show the image when one is chosen
     */
    changed() {
        this.data = null;
        if (this.node !== undefined) {
            let nodeParent = this.node.element.parentElement;
            if (!isEmpty(nodeParent)) {
                let loadedImage = nodeParent.querySelector("img");
                if (!isEmpty(loadedImage)) {
                    nodeParent.removeChild(loadedImage);
                }
            }
        }
        return new Promise(async (resolve, reject) => {
            let value = this.getValue();
            if (value instanceof File) {
                let reader = new FileReader();
                reader.addEventListener("load", async () => {
                    let fileType = reader.result.substring(5, reader.result.indexOf(";"));
                    if (!fileType.startsWith("image")) {
                        this.setValue(null);
                        reject("File must be an image.");
                        return;
                    }
                    this.data = reader.result;
                    let image = new ImageView(this.config, this.data);
                    this.node.element.parentElement.insertBefore(await image.render(), this.node.element.parentElement.children[0]);
                    await super.changed();
                    resolve();
                });
                reader.readAsDataURL(value);
            } else {
                await super.changed();
                resolve();
            }
        });
    }
}

export {
    FileInputView,
    ImageFileInputView
};
