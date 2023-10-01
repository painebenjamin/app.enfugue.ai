/** @module nodes/image-editor/image-node.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { View } from "../../view/base.mjs";
import { ScribbleView } from "../../view/scribble.mjs";
import { ImageView, BackgroundImageView } from "../../view/image.mjs";
import { ImageEditorImageNodeOptionsFormView } from "../../forms/enfugue/image-editor.mjs";
import { CompoundNodeView } from "../base.mjs";
import { ImageEditorNodeView } from "./base.mjs";
import { ImageEditorScribbleNodeView } from "./scribble.mjs";

/**
 * Extend the compound node to help manage image merging settings
 */
class ImageEditorCompoundImageNodeView extends CompoundNodeView {
    /**
     * @var bool Hide the header
     */
    static hideHeader = true;

    /**
     * @var int Modify snap size to 8
     */
    static snapSize = 8;

    /**
     * @var string The name to show in the menu
     */
    static nodeTypeName = "Images";

    /**
     * @var bool Enable header flipping
     */
    static canFlipHeader = true;

    /**
     * @var int Modify padding to 8
     */
    static padding = 8;

    /**
     * @var int Modify edge handler tolerance to 8
     */
    static edgeHandlerTolerance = 8;

    /**
     * @var int Increase min height
     */
    static minHeight = 32;

    /**
     * @var int Increase min width
     */
    static minWidth = 32;

    /**
     * @var string Change from 'Close' to 'Remove'
     */
    static closeText = "Remove";

    /**
     * @var array<string> Methods to pass through (when calling from menu)
     */
    static passThroughMethods = [
        "clearMemory", "increaseSize", "decreaseSize",
        "togglePencilShape", "toggleEraser", "rotateClockwise",
        "rotateCounterClockwise", "mirrorHorizontally", "mirrorVertically",
        "toggleOptions"
    ];

    /**
     * On construct, bind pass-through methods.
     */
    constructor(config, editor, name, content, left, top, width, height) {
        super(config, editor, name, content, left, top, width, height);
        for (let methodName of this.constructor.passThroughMethods) {
            this[methodName] = function () {
                return this.content.selectedNode[methodName].apply(
                    this.content.selectedNode,
                    Array.from(arguments)
                );
            }
        }
    }
}

/**
 * A small class containing the scribble and image
 */
class ImageScribbleView extends View {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-image-scribble-view";

    /**
     * On construct, add sub views
     */
    constructor(config, src, width, height) {
        super(config);
        this.src = src;
        if (!isEmpty(src)) {
            this.image = new BackgroundImageView(config, src);
        }
        this.scribble = new ScribbleView(config, width, height)
        this.clearScribble();
    }

    /**
     * @return bool If the scribble view is erasing
     */
    get isEraser() {
        return this.scribble.isEraser;
    }

    /**
     * @param bool If the scribble view is erasing
     */
    set isEraser(newIsEraser) {
        this.scribble.isEraser = newIsEraser;
    }

    /**
     * @return string The shape of the scribble tool
     */
    get shape() {
        return this.scribble.shape;
    }

    /**
     * @param string The shape of the scribble tool
     */
    set shape(newShape) {
        this.scribble.shape = newShape;
    }

    /**
     * Sets the scribble to an image source, then resizes
     */
    setScribble(source, width, height) {
        this.scribble.setMemory(source);
        this.scribble.resizeCanvas(width, height);
        this.scribble.show();
    }

    /**
     * Clears the scribble memory and hides it
     */
    clearScribble() {
        this.scribble.clearMemory();
        this.scribble.hide();
    }

    /**
     * Clears the scribble memory
     */
    clearMemory(){
        this.scribble.clearMemory();
    }

    /**
     * Increase the scribble size
     */
    increaseSize() {
        this.scribble.increaseSize();
    }

    /**
     * Decrease the scribble size
     */
    decreaseSize() {
        this.scribble.decreaseSize();
    }

    /**
     * Shows the scribble
     */
    showScribble() {
        this.scribble.show();
    }

    /**
     * Resizes the scribble canvas
     */
    resize(width, height) {
        this.scribble.resizeCanvas(width, height);
    }

    /**
     * @return string The data URI of the scribble
     */
    get scribbleSrc() {
        return this.scribble.src;
    }

    /**
     * @return string The data URI or source of the imgae
     */
    get imageSrc() {
        return isEmpty(this.image)
            ? this.src
            : this.image.src;
    }

    /**
     * Mirrors the image horizontally
     */
    mirrorHorizontally() {
        if (!isEmpty(this.image)) {
            return this.image.mirrorHorizontally();
        }
    }

    /**
     * Mirrors the image vertically
     */
    mirrorVertically() {
        if (!isEmpty(this.image)) {
            return this.image.mirrorVertically();
        }
    }

    /**
     * Rotates the image clockwise by 90 degrees
     */
    rotateClockwise() {
        if (!isEmpty(this.image)) {
            return this.image.rotateClockwise();
        }
    }

    /**
     * Rotates the image counter-clockwise by 90 degrees
     */
    rotateCounterClockwise() {
        if (!isEmpty(this.image)) {
            return this.image.rotateCounterClockwise();
        }
    }

    /**
     * Adds a class to the image node
     */
    addImageClass(className) {
        if (!isEmpty(this.image)) {
            this.image.addClass(className);
        }
    }

    /**
     * Removes a class from the image node
     */
    removeImageClass(className) {
        if (!isEmpty(this.image)) {
            this.image.removeClass(className);
        }
    }

    /**
     * On build, append child views
     */
    async build() {
        let node = await super.build();
        node.content(await this.scribble.getNode());
        if (!isEmpty(this.image)) {
            node.append(await this.image.getNode());
        }
        return node;
    }
}

/**
 * When pasting images on the image editor, allow a few fit options
 */
class ImageEditorImageNodeView extends ImageEditorNodeView {
    /**
     * @var bool Hide this header
     */
    static hideHeader = true;

    /**
     * @var bool Enable merging
     */
    static canMerge = true;

    /**
     * @var class A class to help manage merging images
     */
    static compoundNodeClass = ImageEditorCompoundImageNodeView;

    /**
     * @var string The name to show in the menu
     */
    static nodeTypeName = "Image";

    /**
     * @var array<string> All fit modes.
     */
    static allFitModes = ["actual", "stretch", "cover", "contain"];

    /**
     * @var array<string> All anchor modes.
     */
    static allAnchorModes = [
        "top-left", "top-center", "top-right",
        "center-left", "center-center", "center-right",
        "bottom-left", "bottom-center", "bottom-right"
    ];

    /**
     * @var array<string> The node buttons that pertain to scribble.
     */
    static scribbleButtons = [
        "erase", "shape", "clear", "increase", "decrease"
    ];
    
    /**
     * @var string Add the classname for CSS
     */
    static className = 'image-editor-image-node-view';

    /**
     * @var object Buttons to control the scribble. Shortcuts are registered on the view itself.
     */
    static nodeButtons = {
        ...ImageEditorNodeView.nodeButtons,
        ...{
            "shape": {"disabled": true, ...ImageEditorScribbleNodeView.nodeButtons.shape},
            "erase": {"disabled": true, ...ImageEditorScribbleNodeView.nodeButtons.erase},
            "clear": {"disabled": true, ...ImageEditorScribbleNodeView.nodeButtons.clear},
            "increase": {"disabled": true, ...ImageEditorScribbleNodeView.nodeButtons.increase},
            "decrease": {"disabled": true, ...ImageEditorScribbleNodeView.nodeButtons.decrease},
            "mirror-x": {
                "icon": "fa-solid fa-left-right",
                "tooltip": "Mirror Horizontally",
                "shortcut": "z",
                "callback": function() {
                    this.mirrorHorizontally();
                }
            },
            "mirror-y": {
                "icon": "fa-solid fa-up-down",
                "tooltip": "Mirror Vertically",
                "shortcut": "y",
                "callback": function() {
                    this.mirrorVertically();
                }
            },
            "rotate-clockwise": {
                "icon": "fa-solid fa-rotate-right",
                "tooltip": "Rotate Clockwise",
                "shortcut": "r",
                "callback": function() {
                    this.rotateClockwise();
                }
            },
            "rotate-counter-clockwise": {
                "icon": "fa-solid fa-rotate-left",
                "tooltip": "Rotate Counter-Clockwise",
                "shortcut": "w",
                "callback": function() {
                    this.rotateCounterClockwise();
                }
            }
        }
    };

    /**
     * @var class The form for this node.
     */
    static optionsFormView = ImageEditorImageNodeOptionsFormView;

    /**
     * Intercept the constructor to set the contents to use view container.
     */
    constructor(editor, name, content, left, top, width, height) {
        super(
            editor,
            name,
            new ImageScribbleView(
                editor.config,
                isEmpty(content) ? null : content.src,
                width,
                height
            ),
            left,
            top,
            width,
            height
        );
    }

    /**
     * On resize, resize the content as well
     */
    async resized() {
        await super.resized();
        this.content.resize(
            this.visibleWidth - this.constructor.padding * 2,
            this.visibleHeight - this.constructor.padding * 2
        );
    }

    /**
     * Clears the content memory
     */
    clearMemory(){
        this.content.clearMemory();
    }

    /**
     * Increase the content size
     */
    increaseSize() {
        this.content.increaseSize();
    }

    /**
     * Decrease the content size
     */
    decreaseSize() {
        this.content.decreaseSize();
    }

    /**
     * Updates the options after a user makes a change.
     */
    async updateOptions(newOptions) {
        super.updateOptions(newOptions);

        // Reflected in DOM
        this.updateFit(newOptions.fit);
        this.updateAnchor(newOptions.anchor);
        
        // Flags
        this.infer = newOptions.infer;
        this.control = newOptions.control;
        this.inpaint = newOptions.inpaint;
        this.imagePrompt = newOptions.imagePrompt;

        // Conditional inputs
        this.strength = newOptions.strength;
        this.imagePromptScale = newOptions.imagePromptScale;
        this.imagePromptPlus = newOptions.imagePromptPlus;
        this.imagePromptFace = newOptions.imagePromptFace;
        this.controlnet = newOptions.controlnet;
        this.conditioningScale = newOptions.conditioningScale;
        this.conditioningStart = newOptions.conditioningStart;
        this.conditioningEnd = newOptions.conditioningEnd;
        this.processControlImage = newOptions.processControlImage;
        this.invertControlImage = newOptions.invertControlImage;
        this.cropInpaint = newOptions.cropInpaint;
        this.inpaintFeather = newOptions.inpaintFeather;

        // Update scribble view if inpainting
        if (this.node !== undefined) {
            if (this.inpaint) {
                this.content.showScribble();
                // Make sure the scribble view is the right size
                this.content.resize(this.w, this.h);
            } else {
                this.content.clearScribble();
            }
        }

        // Buttons
        if (!isEmpty(this.buttons)) {
            for (let button of this.constructor.scribbleButtons) {
                this.buttons[button].disabled = !newOptions.inpaint;
            }
        }

        this.rebuildHeaderButtons();
    };

    /**
     * Updates the image fit
     */
    async updateFit(newFit) {
        this.fit = newFit;
        this.content.fit = newFit;
        for (let fitMode of this.constructor.allFitModes) {
            this.content.removeImageClass(`fit-${fitMode}`);
        }
        if (!isEmpty(newFit)) {
            this.content.addImageClass(`fit-${newFit}`);
        }
    };

    /**
     * Updates the image anchor
     */
    async updateAnchor(newAnchor) {
        this.anchor = newAnchor;
        this.content.anchor = newAnchor;
        for (let anchorMode of this.constructor.allAnchorModes) {
            this.content.removeImageClass(`anchor-${anchorMode}`);
        }
        if (!isEmpty(newAnchor)) {
            this.content.addImageClass(`anchor-${newAnchor}`);
        }
    }
    
    /**
     * Mirrors the image horizontally
     */
    mirrorHorizontally() {
        return this.content.mirrorHorizontally();
    }
    
    /**
     * Mirrors the image vertically
     */
    mirrorVertically() {
        return this.content.mirrorVertically();
    }
    
    /**
     * Rotates the image clockwise by 90 degrees
     */
    rotateClockwise() {
        return this.content.rotateClockwise();
    }
    
    /**
     * Rotates the image counter-clockwise by 90 degrees
     */
    rotateCounterClockwise() {
        return this.content.rotateCounterClockwise();
    }

    /**
     * Toggle the shape of the scribble pencil
     */
    togglePencilShape() {
        let currentShape = this.content.shape;

        if (currentShape === "circle") {
            this.content.shape = "square";
            this.buttons.shape.tooltip = ImageEditorScribbleNodeView.pencilCircleTooltip;
            this.buttons.shape.icon = ImageEditorScribbleNodeView.pencilCircleIcon;
        } else {
            this.content.shape = "circle";
            this.buttons.shape.tooltip = ImageEditorScribbleNodeView.pencilSquareTooltip;
            this.buttons.shape.icon = ImageEditorScribbleNodeView.pencilSquareIcon;
        }

        this.rebuildHeaderButtons();
    };

    /**
     * Toggles erase mode
     */
    toggleEraser() {
        let currentEraser = this.content.isEraser === true;

        if (currentEraser) {
            this.content.isEraser = false;
            this.buttons.erase.icon = ImageEditorScribbleNodeView.eraserIcon;
            this.buttons.erase.tooltip = ImageEditorScribbleNodeView.eraserTooltip;
        } else {
            this.content.isEraser = true;
            this.buttons.erase.icon = ImageEditorScribbleNodeView.pencilIcon;
            this.buttons.erase.tooltip = ImageEditorScribbleNodeView.pencilTooltip;
        }

        this.rebuildHeaderButtons();
    };

    /**
     * Override getState to include the image, fit and anchor
     */
    getState(includeImages = true) {
        let state = super.getState(includeImages);
        state.scribbleSrc = includeImages ? this.content.scribbleSrc : null;
        state.src = includeImages ? this.content.imageSrc : null;
        state.anchor = this.anchor || null;
        state.fit = this.fit || null;
        state.infer = this.infer || false;
        state.control = this.control || false;
        state.inpaint = this.inpaint || false;
        state.imagePrompt = this.imagePrompt || false;
        state.imagePromptPlus = this.imagePromptPlus || false;
        state.imagePromptFace = this.imagePromptFace || false;
        state.imagePromptScale = this.imagePromptScale || 0.5;
        state.strength = this.strength || 0.8;
        state.controlnet = this.controlnet || null;
        state.cropInpaint = this.cropInpaint !== false;
        state.inpaintFeather = this.inpaintFeather || 32;
        state.conditioningScale = this.conditioningScale || 1.0;
        state.conditioningStart = this.conditioningStart || 0.0;
        state.conditioningEnd = this.conditioningEnd || 1.0;
        state.processControlImage = this.processControlImage !== false;
        state.invertControlImage = this.invertControlImage === true;
        state.removeBackground = this.removeBackground === true;
        state.scaleToModelSize = this.scaleToModelSize === true;
        return state;
    }

    /**
     * Override setState to add the image and scribble
     */
    async setState(newState) {
        await super.setState(newState);
        await this.setContent(new ImageScribbleView(this.config, newState.src, newState.w, newState.h));
        await this.updateAnchor(newState.anchor);
        await this.updateFit(newState.fit);
        if (newState.inpaint) {
            let scribbleImage = new Image();
            scribbleImage.onload = () => {
                this.content.setScribble(scribbleImage, this.w, this.h);
            };
            scribbleImage.src = newState.scribbleSrc;
        } else {
            this.content.clearScribble();
        }
        if (!isEmpty(this.buttons)) {
            for (let button of this.constructor.scribbleButtons) {
                this.buttons[button].disabled = !newState.inpaint;
            }
        }
        this.rebuildHeaderButtons();
    }

    /**
     * Provide a default state for when we are initializing from an image
     */
    static getDefaultState() {
        return {
            "classname": this.name,
            "inpaint": false,
            "control": false,
            "inpaint": false,
            "imagePrompt": false,
            "imagePromptPlus": false,
            "imagePromptFace": false,
            "cropInpaint": true,
            "inpaintFeather": 32,
            "inferenceSteps": null,
            "guidanceScale": null,
            "imagePromptScale": 0.5,
            "strength": 0.8,
            "processControlImage": true,
            "invertControlImage": false,
            "conditioningScale": 1.0,
            "conditioningStart": 0.0,
            "conditioningEnd": 1.0,
            "removeBackground": false,
            "scaleToModelSize": false,
        };
    }

    /**
     * Catch on build to ensure buttons are correct
     */
    async build() {
        let node = await super.build();
        if (this.inpaint === true) {
            for (let button of this.constructor.scribbleButtons) {
                this.buttons[button].disabled = false;
            }
            setTimeout(() => this.rebuildHeaderButtons(), 250);
        }
        return node;
    }
};

export {
    ImageEditorImageNodeView,
    ImageEditorCompoundImageNodeView
};
