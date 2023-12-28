/** @module nodes/image-editor/image.mjs */
import { isEmpty, promptFiles } from "../../base/helpers.mjs";
import { View } from "../../view/base.mjs";
import { ImageView, BackgroundImageView } from "../../view/image.mjs";
import { ImageEditorNodeView } from "./base.mjs";
import { NoImageView } from "./common.mjs";

/**
 * When pasting images on the image editor, allow a few fit options
 */
class ImageEditorImageNodeView extends ImageEditorNodeView {
    /**
     * @var bool Hide header (position absolutely)
     */
    static hideHeader = true;

    /**
     * @var string The name to show in the menu
     */
    static nodeTypeName = "Image";

    /**
     * @var string Add the classname for CSS
     */
    static className = 'image-editor-image-node-view';

    /**
     * @var object Buttons to control the image.
     */
    static nodeButtons = {
        ...ImageEditorNodeView.nodeButtons,
        ...{
            "replace-image": {
                "icon": "fa-solid fa-upload",
                "tooltip": "Replace Image",
                "shortcut": "c",
                "callback": function() {
                    this.replaceImage();
                }
            },
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
     * Updates the options after a user makes a change.
     */
    async updateOptions(newOptions) {
        // Reflected in DOM
        this.updateFit(newOptions.fit);
        this.updateAnchor(newOptions.anchor, newOptions.offsetX, newOptions.offsetY);
        this.updateOpacity(newOptions.opacity);
    };

    /**
     * Updates the image fit
     */
    async updateFit(newFit) {
        this.fit = newFit;
        this.content.setFit(newFit);
    };

    /**
     * Updates the image anchor
     */
    async updateAnchor(newAnchor, offsetX, offsetY) {
        this.anchor = newAnchor;
        this.offsetX = isEmpty(offsetX) ? 0 : offsetX;
        this.offsetY = isEmpty(offsetY) ? 0 : offsetY;
        this.content.setAnchor(this.anchor, this.offsetX, this.offsetY);
    }

    /**
     * Updates the opacity
     */
    async updateOpacity(newOpacity) {
        this.opacity = newOpacity;
        (await this.content.getNode()).css("opacity", newOpacity);
    }

    /**
     * Prompts for a new image
     */
    async replaceImage() {
        let imageToLoad;
        try {
            imageToLoad = await promptFiles("image/*");
        } catch(e) {
            // No files selected
        }
        if (!isEmpty(imageToLoad)) {
            let reader = new FileReader();
            reader.addEventListener("load", async () => {
                let imageView = new BackgroundImageView(this.config, reader.result, false);
                await this.setContent(imageView);
                this.updateFit(this.fit);
                this.updateAnchor(this.anchor);
                this.updateOpacity(this.opacity);
            });
            reader.readAsDataURL(imageToLoad);
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
     * Override getState to include the image, fit and anchor
     */
    getState(includeImages = true) {
        let state = super.getState(includeImages);
        state.src = includeImages ? this.content.src : null;
        state.anchor = this.anchor || null;
        state.offsetX = this.offsetX || null;
        state.offsetY = this.offsetY || null;
        state.fit = this.fit || null;
        state.opacity = this.opacity || null;
        return state;
    }

    /**
     * Override setState to add the image
     */
    async setState(newState) {
        await super.setState(newState);
        if (isEmpty(newState.src)) {
            await this.setContent(new NoImageView(this.config));
        } else {
            await this.setContent(new BackgroundImageView(this.config, newState.src, false));
        }
        await this.updateFit(newState.fit);
        await this.updateAnchor(newState.anchor, newState.offsetX, newState.offsetY);
        await this.updateOpacity(newState.opacity);
    }

    /**
     * Gets the size of the image when scaling the node
     */
    async getCanvasScaleSize() {
        if (isEmpty(this.content.src)) {
            return await super.getCanvasScaleSize();
        } else {
            await this.content.waitForLoad();
            return [
                Math.floor(this.content.width / 8) * 8,
                Math.floor(this.content.height / 8) * 8
            ];
        }
    }

    /**
     * Provide a default state for when we are initializing from an image
     */
    static getDefaultState() {
        return {
            "classname": this.name,
        };
    }
};

export { ImageEditorImageNodeView };
