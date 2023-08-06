/** @module view/nodes/image-editor */
import { isEmpty, filterEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { NodeEditorView } from "./editor.mjs";
import { ImageView } from "../image.mjs";
import { ImageEditorNodeView } from "./image-editor/base.mjs";
import { ImageEditorScribbleNodeView } from "./image-editor/scribble.mjs";
import { ImageEditorPromptNodeView } from "./image-editor/prompt.mjs";
import { ImageEditorImageNodeView } from "./image-editor/image.mjs";
import { CurrentInvocationImageView } from "./image-editor/invocation.mjs";

const E = new ElementBuilder();

/**
 * The ImageEditorView references classes.
 * It also manages to invocation view to view results.
 */
class ImageEditorView extends NodeEditorView {
    /**
     * Image editor contains ref to app.
     */
    constructor(application) {
        super(application.config, window.innerWidth-300, window.innerHeight-70);
        this.application = application;
    }

    /**
     * @var int The width of the canvas (can be changed)
     */
    static canvasWidth = 512;

    /**
     * @var int The height of the canvas (can be changed)
     */
    static canvasHeight = 512;

    /**
     * @var bool Center the editor
     */
    static centered = true;

    /**
     * @var string Add the classname for CSS
     */
    static className = 'image-editor';

    /**
     * @var int Increase the maximum zoom by a lot
     */
    static maximumZoom = 10;

    /**
     * @var array<class> The node classes for state set/get
     */
    static nodeClasses = [
        ImageEditorScribbleNodeView,
        ImageEditorImageNodeView,
        ImageEditorPromptNodeView
    ];

    /**
     * Removes the current invocation from the canvas view.
     */
    hideCurrentInvocation() {
        this.currentInvocation.hide();
        this.removeClass("has-image");
        if (!isEmpty(this.configuredWidth) && !isEmpty(this.configuredHeight)) {
            this.width = this.configuredWidth;
            this.height = this.configuredHeight;
            this.configuredHeight = null;
            this.configuredWidth = null;
        }
    }

    /**
     * Sets a current invocation on the canvas view.
     * @param string $href The image source.
     */
    async setCurrentInvocationImage(href) {
        this.currentInvocation.setImage(href);
        await this.currentInvocation.waitForLoad();
        if (this.currentInvocation.width != this.width || this.currentInvocation.height != this.height) {
            if (isEmpty(this.configuredWidth)) {
                this.configuredWidth = this.width;
            }
            if (isEmpty(this.configuredHeight)) {
                this.configuredHeight = this.height;
            }
            this.width = this.currentInvocation.width;
            this.height = this.currentInvocation.height;
        }
        this.currentInvocation.show();
        this.addClass("has-image");
    }

    /**
     * Gets the next unoccupied [x, y]
     */
    getNextNodePoint() {
        let nodeX = this.nodes.map((node) => node.left + ImageEditorNodeView.padding),
            nodeY = this.nodes.map((node) => node.top + ImageEditorNodeView.padding),
            [x, y] = [0, 0];
        
        while (nodeX.indexOf(x) !== -1) x += ImageEditorNodeView.snapSize;
        while (nodeY.indexOf(y) !== -1) y += ImageEditorNodeView.snapSize;
        return [x, y];
    }

    /**
     * This is a shorthand helper functinon for adding an image node.
     * @param string $imageSource The source of the image - likely a data URL.
     * @return NodeView The added view.
     */
    async addImageNode(imageSource, imageName = "Image") {
        let imageView = new ImageView(this.config, imageSource),
            [x, y] = this.getNextNodePoint();
        await imageView.waitForLoad();
        return await this.addNode(
            ImageEditorImageNodeView,
            imageName,
            imageView,
            x,
            y,
            imageView.width,
            imageView.height
        );
    }

    /**
     * This is a shorthand helper for adding a scribble node.
     * @return NodeView The added view
     */
    async addScribbleNode(scribbleName = "Scribble") {
        let [x, y] = this.getNextNodePoint();
        return await this.addNode(
            ImageEditorScribbleNodeView,
            scribbleName,
            null,
            x,
            y,
            256,
            256
        );
    }
    
    /**
     * This is a shorthand helper for adding a prompt node.
     * @return NodeView The added view.
     */
    async addPromptNode(promptName = "Prompt") {
        let [x, y] = this.getNextNodePoint();
        return await this.addNode(
            ImageEditorPromptNodeView,
            promptName,
            null,
            x,
            y,
            256,
            256
        );
    }

    /**
     * Builds the DOMElement
     */
    async build() {
        let node = await super.build(),
            grid = E.createElement("enfugue-image-editor-grid");
        this.currentInvocation = new CurrentInvocationImageView(this);
        this.currentInvocation.hide();
        node.find("enfugue-node-canvas").append(grid, await this.currentInvocation.getNode());
        return node;
    }

    /**
     * Gets base state when initializing from an image
     */
    static getNodeDataForImage(image) {
        let baseState = {
            "x": 0,
            "y": 0,
            "w": image.width,
            "h": image.height,
            "src": image.src,
            "name": "Initial Image"
        };
        return {...baseState, ...ImageEditorImageNodeView.getDefaultState()};        
    }
}

export { 
    ImageEditorView,
    /* Re-exports */
    ImageEditorNodeView,
    ImageEditorImageNodeView,
    ImageEditorScribbleNodeView,
    ImageEditorPromptNodeView
};
