/** @module nodes/image-editor/invocation.mjs */
import { SimpleNotification } from "../../common/notify.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { ImageView } from "../../view/image.mjs";
import { ToolbarView } from "../../view/menu.mjs";
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
     * Starts filtering the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageFilter() {
        if (!isEmpty(this.imageFilterWindow)) {
            return this.imageFilterWindow.focus();
        }
        if (!isEmpty(this.imageAdjustmentWindow)) {
            return this.editor.application.notifications.push(
                "warn",
                "Can't Filter Right Now",
                "Complete image adjustments before trying to filter."
            );
        }

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
        if (!isEmpty(this.imageAdjustmentWindow)) {
            return this.imageAdjustmentWindow.focus();
        }
        if (!isEmpty(this.imageFilterWindow)) {
            return this.editor.application.notifications.push(
                "warn",
                "Can't Adjust Right Now",
                "Complete image filters before trying to adjust."
            );
        }
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
