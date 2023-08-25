/** @module nodes/image-editor/prompt.mjs */
import { ImageEditorNodeView } from "./base.mjs";
import { ImageEditorNodeOptionsFormView } from "../../forms/enfugue/image-editor.mjs";

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
        let realContent = new ImageEditorNodeOptionsFormView(editor.config);
        super(editor, name, realContent, left, top, width, height);
        realContent.onSubmit((values) => this.updateOptions(values));
    }

    /**
     * Gets state from the content
     */
    getState(includeImages = true) {
        let state = super.getState(includeImages);
        state = {...state, ...this.content.values};
        return state;
    }

    /**
     * Set the state on the content
     */
    async setState(newState) {
        await super.setState(newState);
        await this.content.getNode(); // Wait for first build
        await this.content.setValues(newState);
    }
};

export { ImageEditorPromptNodeView };
