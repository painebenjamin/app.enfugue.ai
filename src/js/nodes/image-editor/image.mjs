/** @module nodes/image-editor/image-node.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { ImageView, BackgroundImageView } from "../../view/image.mjs";
import { ImageEditorImageNodeOptionsFormView } from "../../forms/enfugue/image-editor.mjs";
import { ImageEditorScribbleNodeView } from "./scribble.mjs";

/**
 * When pasting images on the image editor, allow a few fit options
 */
class ImageEditorImageNodeView extends ImageEditorScribbleNodeView {
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
        ...ImageEditorScribbleNodeView.nodeButtons,
        ...{
            "mirror-x": {
                "icon": "fa-solid fa-left-right",
                "tooltip": "Mirror Horizontally",
                "shortcut": "h",
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
     * Intercept the constructor to set the contents to use background image instead of base image.
     */
    constructor(editor, name, content, left, top, width, height) {
        if (content instanceof ImageView) {
            content = new BackgroundImageView(content.config, content.src);
        }
        super(editor, name, null, left, top, width, height);
        this.scribbleView = this.content;
        this.content = content;
        // Hide scribbble view 
        this.scribbleView.hide();
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

        // Conditional inputs
        this.strength = newOptions.strength;
        this.controlnet = newOptions.controlnet;
        this.conditioningScale = newOptions.conditioningScale;
        this.processControlImage = newOptions.processControlImage;
        this.cropInpaint = newOptions.cropInpaint;
        this.inpaintFeather = newOptions.inpaintFeather;

        // Update scribble view if inpainting
        if (this.node !== undefined) {
            let nodeHeader = this.node.find("enfugue-node-header");
            if (this.inpaint) {
                this.scribbleView.show();
                // Make sure the scribble view is the right size
                this.scribbleView.resizeCanvas(this.w, this.h);
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).show();
                }
            } else {
                this.scribbleView.hide();
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).hide();
                }
            }
        }
    };

    /**
     * Updates the image fit
     */
    async updateFit(newFit) {
        this.fit = newFit;
        let content = await this.getContent();
        content.fit = newFit;
        for (let fitMode of this.constructor.allFitModes) {
            content.removeClass(`fit-${fitMode}`);
        }
        if (!isEmpty(newFit)) {
            content.addClass(`fit-${newFit}`);
        }
    };

    /**
     * Updates the image anchor
     */
    async updateAnchor(newAnchor) {
        this.anchor = newAnchor;
        let content = await this.getContent();
        for (let anchorMode of this.constructor.allAnchorModes) {
            content.removeClass(`anchor-${anchorMode}`);
        }
        if (!isEmpty(newAnchor)) {
            content.addClass(`anchor-${newAnchor}`);
        }
    }
    
    /**
     * On build, append the scribble view (hidden) and hide scribble buttons
     */
    async build() {
        let node = await super.build();
        node.find("enfugue-node-contents").append(await this.scribbleView.getNode());
        let nodeHeader = node.find("enfugue-node-header");
        for (let button of this.constructor.scribbleButtons) {
            nodeHeader.find(`.node-button-${button}`).hide();
        }
        return node;
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
     * Override getState to include the image, fit and anchor
     */
    getState(includeImages = true) {
        let state = super.getState(includeImages);
        state.scribbleSrc = includeImages ? this.scribbleView.src : null;
        state.src = includeImages ? this.content.src : null;
        state.anchor = this.anchor || null;
        state.fit = this.fit || null;
        state.infer = this.infer || false;
        state.control = this.control || false;
        state.inpaint = this.inpaint || false;
        state.strength = this.strength || 0.8;
        state.controlnet = this.controlnet || null;
        state.colorSpace = this.colorSpace || "invert";
        state.cropInpaint = this.cropInpaint !== false;
        state.inpaintFeather = this.inpaintFeather || 32;
        state.conditioningScale = this.conditioningScale || 1.0;
        state.processControlImage = this.processControlImage !== false;
        state.removeBackground = this.removeBackground === true;
        state.scaleToModelSize = this.scaleToModelSize === true;
        return state;
    }

    /**
     * Override setState to add the image and scribble
     */
    async setState(newState) {
        await this.setContent(new BackgroundImageView(this.config, newState.src));
        await this.updateAnchor(newState.anchor);
        await this.updateFit(newState.fit);
        if (newState.inpaint) {
            let scribbleImage = new Image();
            scribbleImage.onload = () => {
                this.scribbleView.setMemory(scribbleImage);
                this.scribbleView.resizeCanvas(this.w, this.h);
                this.scribbleView.show();
                if (this.node !== undefined) {
                    let nodeHeader = this.node.find("enfugue-node-header");
                    for (let button of this.constructor.scribbleButtons) {
                        nodeHeader.find(`.node-button-${button}`).show();
                    }
                }
            };
            scribbleImage.src = newState.scribbleSrc;
        } else {
            this.scribbleView.clearMemory();
            this.scribbleView.hide();
            if (this.node !== undefined) {
                let nodeHeader = this.node.find("enfugue-node-header");
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).hide();
                }
            }
        }
        await super.setState({...newState, ...{"src": null}});
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
            "cropInpaint": true,
            "inpaintFeather": 32,
            "inferenceSteps": null,
            "guidanceScale": null,
            "strength": 0.8,
            "processControlImage": true,
            "colorSpace": "invert",
            "conditioningScale": 1.0,
            "removeBackground": false,
            "scaleToModelSize": false,
        };
    }
};

export { ImageEditorImageNodeView };
