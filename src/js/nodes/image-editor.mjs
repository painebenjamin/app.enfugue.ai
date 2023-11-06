/** @module nodes/image-editor */
import { isEmpty, filterEmpty } from "../base/helpers.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { NodeEditorView } from "./editor.mjs";
import { ImageView, BackgroundImageView } from "../view/image.mjs";
import { VideoView } from "../view/video.mjs";
import { ImageEditorNodeView } from "./image-editor/base.mjs";
import { ImageEditorScribbleNodeView } from "./image-editor/scribble.mjs";
import { ImageEditorPromptNodeView } from "./image-editor/prompt.mjs";
import { ImageEditorImageNodeView } from "./image-editor/image.mjs";
import { ImageEditorVideoNodeView } from "./image-editor/video.mjs";
import { NoImageView, NoVideoView } from "./image-editor/common.mjs";

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
        ImageEditorVideoNodeView,
    ];

    /**
     * On focus, register/unregister menus
     */
    async focusNode(node) {
        super.focusNode(node);
        this.focusedNode = node;
        this.application.menu.removeCategory("Node");
        let nodeButtons = node.getButtons();
        if (!isEmpty(nodeButtons)) {
            let menuCategory = await this.application.menu.addCategory("Node", "n");
            for (let buttonName in nodeButtons) {
                let buttonConfiguration = nodeButtons[buttonName];
                let menuItem = await menuCategory.addItem(
                    buttonConfiguration.tooltip,
                    buttonConfiguration.icon,
                    buttonConfiguration.shortcut
                );
                menuItem.onClick(() => buttonConfiguration.callback.call(node));
            }
        }
    }

    /**
     * On remove, check if we need to remove the menu
     */
    async removeNode(node) {
        super.removeNode(node);
        if (this.focusedNode === node) {
            this.focusedNode = null;
            this.application.menu.removeCategory("Node");
        }
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
        let imageView = null,
            [x, y] = this.getNextNodePoint();

        if (imageSource instanceof ImageView) {
            imageView = imageSource;
        } else if (!isEmpty(imageSource)) {
            imageView = new BackgroundImageView(this.config, imageSource, false);
        } else {
            imageView = new NoImageView(this.config);
        }

        if (imageView instanceof ImageView) {
            await imageView.waitForLoad();
        }

        let newNode = await this.addNode(
            ImageEditorImageNodeView,
            this.getUniqueNodeName(imageName),
            imageView,
            x,
            y,
            imageView.width,
            imageView.height
        );

        return newNode;
    }

    /**
     * This is a shorthand helper functinon for adding a video URL.
     * @param string $videoSource The source of the video - likely a data URL.
     * @return NodeView The added view.
     */
    async addVideoNode(videoSource, videoName = "Video") {
        let videoView = null,
            [x, y] = this.getNextNodePoint();

        if (videoSource instanceof VideoView) {
            videoView = videoSource;
        } else if (!isEmpty(videoSource)) {
            videoView = new VideoView(this.config, videoSource, false);
        } else {
            videoView = new NoVideoView(this.config);
        }

        if (videoView instanceof VideoView) {
            await videoView.waitForLoad();
        }

        let newNode = await this.addNode(
            ImageEditorVideoNodeView,
            this.getUniqueNodeName(videoName),
            videoView,
            x,
            y,
            videoView.width,
            videoView.height
        );

        return newNode;
    }


    /**
     * This is a shorthand helper for adding a scribble node.
     * @return NodeView The added view
     */
    async addScribbleNode(scribbleName = "Scribble") {
        let [x, y] = this.getNextNodePoint();
        return await this.addNode(
            ImageEditorScribbleNodeView,
            this.getUniqueNodeName(scribbleName),
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
            overlays = E.createElement("enfugue-image-editor-overlay"),
            grid = E.createElement("enfugue-image-editor-grid");
        node.find("enfugue-node-canvas").append(overlays.content(grid));
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
        return {...ImageEditorImageNodeView.getDefaultState(), ...baseState};
    }
}

export { 
    ImageEditorView,
    /* Re-exports */
    ImageEditorNodeView,
    ImageEditorImageNodeView,
    ImageEditorScribbleNodeView,
};
