/** @module nodes/image-editor/prompt.mjs */
import { ElementBuilder } from "../../base/builder.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { View } from "../../view/base.mjs";
import { ImageEditorNodeView } from "./base.mjs";

const E = new ElementBuilder();

class PromptNodeContentView extends View {
    /**
     * @var string tag name
     */
    static tagName = "enfugue-region-prompts";

    /**
     * @var string The text to display initially
     */
    static placeholderText = "Use the layer options menu to add a prompt. This region will be filled with an image generated from that prompt, <em>instead</em> of the global prompt.<br /><br />Any remaining empty regions will be inpainted.<br /><br />Check <strong>Remove Background</strong> to remove the background before merging down and inpainting.";

    /**
     * Sets the prompts
     */
    setPrompts(positive, negative = null) {
        if (isEmpty(positive)) {
            this.node.content(E.p().content(this.constructor.placeholderText));
        } else {
            let positiveContent = Array.isArray(positive)
                ? positive.join(", ")
                : positive;
            let contentArray = [E.p().content(positiveContent)];
            if (!isEmpty(negative)) {
                let negativeContent = Array.isArray(negative)
                    ? negative.join(", ")
                    : negative;
                contentArray.push(E.p().content(negativeContent));
            }
            this.node.content(...contentArray);
        }
    }
    
    /**
     * On first build, append placeholder
     */
    async build() {
        let node = await super.build();
        node.content(E.p().content(this.constructor.placeholderText));
        return node;
    }
}

/**
 * The PromptNode just allows for regions to have different prompts.
 */
class ImageEditorPromptNodeView extends ImageEditorNodeView {
    /**
     * @var bool Disable header hiding
     */
    static hideHeader = false;
    
    /**
     * @var bool Disable header flipping
     */
    static canFlipHeader = false;

    /**
     * @var string The default node name
     */
    static nodeName = "Prompt";

    /**
     * @var object Remove other buttons
     */
    static nodeButtons = {};

    /**
     * @var string The classname of the node
     */
    static className = "image-editor-prompt-node-view";

    /**
     * Intercept the constructor to set the contents to the options.
     */
    constructor(editor, name, content, left, top, width, height) {
        let realContent = new PromptNodeContentView(editor.config);
        super(editor, name, realContent, left, top, width, height);
    }

    /**
     * Sets the prompts
     */
    setPrompts(positive, negative = null) {
        this.content.setPrompts(positive, negative);
    }
};

export { ImageEditorPromptNodeView };
