/** @module nodes/image-editor/invocation.mjs */
import { SimpleNotification } from "../../common/notify.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { ImageView } from "../../view/image.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import {
    QuickUpscaleFormView,
    QuickDownscaleFormView
} from "../../forms/enfugue/upscale.mjs";
import {
    ImageAdjustmentView,
    ImageFilterView
} from "./filter.mjs";

/**
 * Extend the ToolbarView slightly to add mouse enter event listeners
 */
class InvocationToolbarView extends ToolbarView {
    constructor(invocationNode) {
        super(invocationNode.config);
        this.invocationNode = invocationNode;
    }
    
    /**
     * onMouseEnter, trigger parent onMouseEnter
     */
    async onMouseEnter(e) {
        this.invocationNode.toolbarEntered();
    }

    /**
     * onMouseLeave, trigger parent onMouseLeave
     */
    async onMouseLeave(e) {
        this.invocationNode.toolbarLeft();
    }

    /**
     * On build, bind events
     */
    async build() {
        let node = await super.build();
        node.on("mouseenter", (e) => this.onMouseEnter(e));
        node.on("mouseleave", (e) => this.onMouseLeave(e));
        return node;
    }
};

/**
 * Create a small extension of the ImageView to change the class name for CSS.
 */
class CurrentInvocationImageView extends ImageView {
    /**
     * Constructed by the editor, pass reference so we can call other functions
     */
    constructor(editor) {
        super(editor.config);
        this.editor = editor;
    }

    /**
     * @var string The class name to apply to the image node
     */
    static className = "current-invocation-image-view";

    /**
     * @var int The number of milliseconds to wait after leaving the image to hide tools
     */
    static hideTime = 250;

    /**
     * @var int The width of the adjustment window in pixels
     */
    static imageAdjustmentWindowWidth = 750;
    
    /**
     * @var int The height of the adjustment window in pixels
     */
    static imageAdjustmentWindowHeight = 525;
    
    /**
     * @var int The width of the filter window in pixels
     */
    static imageFilterWindowWidth = 450;
    
    /**
     * @var int The height of the filter window in pixels
     */
    static imageFilterWindowHeight = 350;

    /**
     * @var int The width of the upscale window in pixels
     */
    static imageUpscaleWindowWidth = 260;

    /**
     * @var int The height of the upscale window in pixels
     */
    static imageUpscaleWindowHeight = 210;

    /**
     * @var int The width of the upscale window in pixels
     */
    static imageDownscaleWindowWidth = 260;

    /**
     * @var int The height of the upscale window in pixels
     */
    static imageDownscaleWindowHeight = 210;

    /**
     * Gets the toolbar node, building if needed
     */
    async getTools() {
        if (isEmpty(this.toolbar)) {
            this.toolbar = new InvocationToolbarView(this);
            
            this.hideImage = await this.toolbar.addItem("Hide Image", "fa-solid fa-eye-slash");
            this.hideImage.onClick(() => this.editor.application.images.hideCurrentInvocation());

            if (!!navigator.clipboard && typeof ClipboardItem === "function") {
                this.copyImage = await this.toolbar.addItem("Copy to Clipboard", "fa-solid fa-clipboard");
                this.copyImage.onClick(() => this.copyToClipboard());
            }

            this.popoutImage = await this.toolbar.addItem("Popout Image", "fa-solid fa-arrow-up-right-from-square");
            this.popoutImage.onClick(() => this.sendToWindow());

            this.saveImage = await this.toolbar.addItem("Save As", "fa-solid fa-floppy-disk");
            this.saveImage.onClick(() => this.saveToDisk());

            this.adjustImage = await this.toolbar.addItem("Adjust Image", "fa-solid fa-sliders");
            this.adjustImage.onClick(() => this.startImageAdjustment());

            this.filterImage = await this.toolbar.addItem("Filter Image", "fa-solid fa-wand-magic-sparkles");
            this.filterImage.onClick(() => this.startImageFilter());

            this.editImage = await this.toolbar.addItem("Edit Image", "fa-solid fa-pen-to-square");
            this.editImage.onClick(() => this.sendToCanvas());

            this.upscaleImage = await this.toolbar.addItem("Upscale Image", "fa-solid fa-up-right-and-down-left-from-center");
            this.upscaleImage.onClick(() => this.startImageUpscale());

            this.downscaleImage = await this.toolbar.addItem("Downscale Image", "fa-solid fa-down-left-and-up-right-to-center");
            this.downscaleImage.onClick(() => this.startImageDownscale());
        }
        return this.toolbar;
    }

    /**
     * Adds the image menu to the passed menu
     */
    async prepareMenu(menu) {
        let hideImage = await menu.addItem("Hide Image", "fa-solid fa-eye-slash", "d");
        hideImage.onClick(() => this.editor.application.images.hideCurrentInvocation());

        if (!!navigator.clipboard && typeof ClipboardItem === "function") {
            let copyImage = await menu.addItem("Copy to Clipboard", "fa-solid fa-clipboard", "c");
            copyImage.onClick(() => this.copyToClipboard());
        }
        
        let popoutImage = await menu.addItem("Popout Image", "fa-solid fa-arrow-up-right-from-square", "p");
        popoutImage.onClick(() => this.sendToWindow());

        let saveImage = await menu.addItem("Save As", "fa-solid fa-floppy-disk", "a");
        saveImage.onClick(() => this.saveToDisk());

        let adjustImage = await menu.addItem("Adjust Image", "fa-solid fa-sliders", "j");
        adjustImage.onClick(() => this.startImageAdjustment());

        let filterImage = await menu.addItem("Filter Image", "fa-solid fa-wand-magic-sparkles", "l");
        filterImage.onClick(() => this.startImageFilter());

        let editImage = await menu.addItem("Edit Image", "fa-solid fa-pen-to-square", "t");
        editImage.onClick(() => this.sendToCanvas());

        let upscaleImage = await menu.addItem("Upscale Image", "fa-solid fa-up-right-and-down-left-from-center", "u");
        upscaleImage.onClick(() => this.startImageUpscale());

        let downscaleImage = await menu.addItem("Downscale Image", "fa-solid fa-down-left-and-up-right-to-center", "w");
        downscaleImage.onClick(() => this.startImageDownscale());
    }

    /**
     * Override parent setImage to also set the image on the adjustment canvas, if present
     */
    setImage(newImage) {
        super.setImage(newImage);
        if (!isEmpty(this.imageAdjuster)) {
            this.imageAdjuster.setImage(newImage);
        }
    }

    /**
     * Triggers the copy to clipboard
     * Chromium only as of 2023-06-21
     */
    async copyToClipboard() {
        navigator.clipboard.write([
            new ClipboardItem({
                "image/png": await this.getBlob()
            })
        ]);
        SimpleNotification.notify("Copied to clipboard!", 2000);
    }

    /**
     * Saves the image to disk
     * Asks for a filename first
     */
    async saveToDisk() {
        this.editor.application.saveBlobAs("Save Image", await this.getBlob(), ".png");
    }

    /**
     * Sends the image to a new canvas
     * Asks for details regarding additional state when clicked
     */
    async sendToCanvas() {
        this.editor.application.initializeStateFromImage(await this.getImageAsDataURL());
    }

    /**
     * Starts downscaling the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageDownscale() {
        if (this.checkActiveTool("downscale")) return;

        let imageBeforeDownscale = this.src,
            setDownscaleAmount = async (amount) => {
                let image = new ImageView(this.config, imageBeforeDownscale);
                await image.waitForLoad();
                await image.downscale(amount);
                this.setImage(image.src);
                this.editor.width = image.width;
                this.editor.height = image.height;
            },
            saveResults = false;

        this.imageDownscaleForm = new QuickDownscaleFormView(this.config);
        this.imageDownscaleWindow = await this.editor.application.windows.spawnWindow(
            "Downscale Image",
            this.imageDownscaleForm,
            this.constructor.imageDownscaleWindowWidth,
            this.constructor.imageDownscaleWindowHeight
        );
        this.imageDownscaleWindow.onClose(() => {
            this.imageDownscaleForm = null;
            this.imageDownscaleWindow = null;
            if (!saveResults) {
                this.setImage(imageBeforeDownscale);
            }
        });
        this.imageDownscaleForm.onChange(async () => {
            let image = new ImageView(this.config, imageBeforeDownscale);
            await image.waitForLoad();
            await image.downscale(this.imageDownscaleForm.values.downscale);
            this.setImage(image.src);
            this.editor.width = image.width;
            this.editor.height = image.height;
        });
        this.imageDownscaleForm.onSubmit(async (values) => {
            saveResults = true;
            // Remove window
            this.imageDownscaleWindow.remove();
        });
        this.imageDownscaleForm.onCancel(() => this.imageDownscaleWindow.remove());
        setDownscaleAmount(2); // Default to 2
    }

    /**
     * Starts upscaling the image
     * Does not replace the current visible canvas.
     * This will use the canvas and upscale settings to send to the backend.
     */
    async startImageUpscale() {
        if (this.checkActiveTool("upscale")) return;

        this.imageUpscaleForm = new QuickUpscaleFormView(this.config);
        this.imageUpscaleWindow = await this.editor.application.windows.spawnWindow(
            "Upscale Image",
            this.imageUpscaleForm,
            this.constructor.imageUpscaleWindowWidth,
            this.constructor.imageUpscaleWindowHeight
        );
        this.imageUpscaleWindow.onClose(() => {
            this.imageUpscaleForm = null;
            this.imageUpscaleWindow = null;
        });
        this.imageUpscaleForm.onCancel(() => this.imageUpscaleWindow.remove());
        this.imageUpscaleForm.onSubmit(async (values) => {
            await this.editor.application.initializeStateFromImage(
                await this.getImageAsDataURL(),
                true, // Save history
                true, // Keep current state, except for...
                {
                    "upscale": {"outscale": values.upscale},
                    "generation": {"samples": 1},
                    "samples": null
                } // ...these state overrides
            );
            // Remove window
            this.imageUpscaleWindow.remove();

            // Hide current invocation
            this.editor.application.images.hideCurrentInvocation()

            // Wait a few ticks then trigger invoke
            setTimeout(() => {
                this.editor.application.publish("tryInvoke");
            }, 2000);
        });
    }

    /**
     * Starts filtering the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageFilter() {
        if (this.checkActiveTool("filter")) return;

        this.imageFilterView = new ImageFilterView(this.config, this.src, this.node.element.parentElement),
        this.imageFilterWindow = await this.editor.application.windows.spawnWindow(
            "Filter Image",
            this.imageFilterView,
            this.constructor.imageFilterWindowWidth,
            this.constructor.imageFilterWindowHeight
        );

        let reset = () => {
            try {
                this.imageFilterView.removeCanvas();
            } catch(e) { }
            this.imageFilterView = null;
            this.imageFilterWindow = null;
        }

        this.imageFilterWindow.onClose(reset);
        this.imageFilterView.onSave(async () => {
            this.setImage(this.imageFilterView.getImageSource());
            setTimeout(() => {
                this.imageFilterWindow.remove();
                reset();
            }, 150);
        });
        this.imageFilterView.onCancel(() => {
            this.imageFilterWindow.remove();
            reset();
        });
    }

    /**
     * Starts adjusting the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageAdjustment() {
        if (this.checkActivetool("adjust")) return;

        this.imageAdjustmentView = new ImageAdjustmentView(this.config, this.src, this.node.element.parentElement),
        this.imageAdjustmentWindow = await this.editor.application.windows.spawnWindow(
            "Adjust Image",
            this.imageAdjustmentView,
            this.constructor.imageAdjustmentWindowWidth,
            this.constructor.imageAdjustmentWindowHeight
        );

        let reset = () => {
            try {
                this.imageAdjustmentView.removeCanvas();
            } catch(e) { }
            this.imageAdjustmentView = null;
            this.imageAdjustmentWindow = null;
        }

        this.imageAdjustmentWindow.onClose(reset);
        this.imageAdjustmentView.onSave(async () => {
            this.setImage(this.imageAdjustmentView.getImageSource());
            setTimeout(() => {
                this.imageAdjustmentWindow.remove();
                reset();
            }, 150);
        });
        this.imageAdjustmentView.onCancel(() => {
            this.imageAdjustmentWindow.remove();
            reset();
        });
    }

    /**
     * Checks if there is an active tool and either:
     * 1. If the active tool matches the intended action, focus on it
     * 2. If the active tool does not, display a warning
     * Then return true. If there is no active tool, return false.
     */
    checkActiveTool(intendedAction) {
        if (!isEmpty(this.imageAdjustmentWindow)) {
            if (intendedAction !== "adjust") {
                this.editor.application.notifications.push(
                    "warn",
                    "Finish Adjusting",
                    `Complete adjustments before trying to ${intendedAction}.`
                );
            } else {
                this.imageAdjustmentWindow.focus();
            }
            return true;
        }
        if (!isEmpty(this.imageFilterWindow)) {
            if (intendedAction !== "filter") {
                this.editor.application.notifications.push(
                    "warn",
                    "Finish Filtering",
                    `Complete filtering before trying to ${intendedAction}.`
                );
            } else {
                this.imageFilterWindow.focus();
            }
            return true;
        }
        if (!isEmpty(this.imageUpscaleWindow)) {
            if (intendedAction !== "upscale") {
                this.editor.application.notifications.push(
                    "warn",
                    "Finish Upscaling",
                    `Complete your upscale selection or cancel before trying to ${intendedAction}.`
                );
            } else {
                this.imageUpscaleWindow.focus();
            }
            return true;
        }
        if (!isEmpty(this.imagedownscaleWindow)) {
            if (intendedAction !== "downscale") {
                this.editor.application.notifications.push(
                    "warn",
                    "Finish Downscaling",
                    `Complete your downscale selection or cancel before trying to ${intendedAction}.`
                );
            } else {
                this.imagedownscaleWindow.focus();
            }
            return true;
        }
        return false;
    }

    /**
     * Opens the image in a new window
     */
    async sendToWindow() {
        window.open(this.src);
    }

    /**
     * The callback when the toolbar has been entered
     */
    async toolbarEntered() {
        this.stopHideTimer();
    }

    /**
     * The callback when the toolbar has been left
     */
    async toolbarLeft() {
        this.startHideTimer();
    }

    /**
     * Stops the timeout that will hide tools
     */
    stopHideTimer() {
        clearTimeout(this.timer);
    }

    /**
     * Start the timeout that will hide tools
     */
    startHideTimer() {
        this.timer = setTimeout(async () => {
            let release = await this.lock.acquire();
            let toolbar = await this.getTools();
            this.node.element.parentElement.removeChild(await toolbar.render());
            release();
        }, this.constructor.hideTime);
    }

    /**
     * The callback for MouseEnter
     */
    async onMouseEnter(e) {
        this.stopHideTimer();
        let release = await this.lock.acquire();
        let toolbar = await this.getTools();
        this.node.element.parentElement.appendChild(await toolbar.render());
        release();
    }

    /**
     * The callback for MouesLeave
     */
    async onMouseLeave(e) {
        this.startHideTimer();
    }

    /**
     * On build, bind mouseenter to show tools
     */
    async build() {
        let node = await super.build();
        node.on("mouseenter", (e) => this.onMouseEnter(e));
        node.on("mouseleave", (e) => this.onMouseLeave(e));
        return node;
    }
};

export {
    InvocationToolbarView,
    CurrentInvocationImageView
};
