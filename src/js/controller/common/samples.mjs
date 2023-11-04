/** @module controller/common/samples */
import { isEmpty, waitFor } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { SimpleNotification } from "../../common/notify.mjs";
import { SampleChooserView } from "../../view/samples/chooser.mjs";
import { SampleView } from "../../view/samples/viewer.mjs";
import {
    ImageAdjustmentView,
    ImageFilterView
} from "../../view/samples/filter.mjs";
import { View } from "../../view/base.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import {
    UpscaleFormView,
    DownscaleFormView
} from "../../forms/enfugue/upscale.mjs";

/**
 * This is the main controller that manages state and views
 */
class SamplesController extends Controller {

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
    static imageUpscaleWindowWidth = 300;

    /**
     * @var int The height of the upscale window in pixels
     */
    static imageUpscaleWindowHeight = 320;

    /**
     * @Xvar int The width of the upscale window in pixels
     */
    static imageDownscaleWindowWidth = 260;

    /**
     * @var int The height of the upscale window in pixels
     */
    static imageDownscaleWindowHeight = 210;

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
     * Pass through some functions to imageview
     */
    async waitForLoad() {
        await this.imageView.waitForLoad();
    }

    /**
     * Triggers the copy to clipboard
     */
    async copyToClipboard() {
        navigator.clipboard.write([
            new ClipboardItem({
                "image/png": await this.imageView.getBlob()
            })
        ]);
        SimpleNotification.notify("Copied to clipboard!", 2000);
    }

    /**
     * Saves the image to disk
     * Asks for a filename first
     */
    async saveToDisk() {
        this.editor.application.saveBlobAs("Save Image", await this.imageView.getBlob(), ".png");
    }

    /**
     * Sends the image to a new canvas
     * Asks for details regarding additional state when clicked
     */
    async sendToCanvas() {
        this.editor.application.initializeStateFromImage(
            await this.imageView.getImageAsDataURL(),
            true, // Save history
            null, // Prompt for current state
            {
                "samples": null
            } // Remove sample chooser
        );
    }

    /**
     * Starts downscaling the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageDownscale() {
        if (this.checkActiveTool("downscale")) return;

        let imageBeforeDownscale = this.imageView.src,
            widthBeforeDownscale = this.imageView.width,
            heightBeforeDownscale = this.imageView.height,
            setDownscaleAmount = async (amount) => {
                let image = new ImageView(this.config, imageBeforeDownscale);
                await image.waitForLoad();
                await image.downscale(amount);
                this.imageView.setImage(image.src);
                this.editor.setDimension(image.width, image.height, false);
            },
            saveResults = false;

        this.imageDownscaleForm = new DownscaleFormView(this.config);
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
                this.imageView.setImage(imageBeforeDownscale);
                this.editor.setDimension(widthBeforeDownscale, heightBeforeDownscale, false);
            }
        });
        this.imageDownscaleForm.onChange(async () => setDownscaleAmount(this.imageDownscaleForm.values.downscale));
        this.imageDownscaleForm.onCancel(() => this.imageDownscaleWindow.remove());
        this.imageDownscaleForm.onSubmit(async (values) => {
            saveResults = true;
            this.imageDownscaleWindow.remove();
        });
        setDownscaleAmount(2); // Default to 2
    }

    /**
     * Starts upscaling the image
     * Does not replace the current visible canvas.
     * This will use the canvas and upscale settings to send to the backend.
     */
    async startImageUpscale() {
        if (this.checkActiveTool("upscale")) return;

        this.imageUpscaleForm = new UpscaleFormView(this.config);
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
                await this.imageView.getImageAsDataURL(),
                true, // Save history
                true, // Keep current state, except for...
                {
                    "upscale": [values],
                    "generation": {"samples": 1, "iterations": 1},
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

        this.imageFilterView = new ImageFilterView(this.config, this.imageView.src, this.node.element.parentElement),
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
            this.imageView.setImage(this.imageFilterView.getImageSource());
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
        if (this.checkActiveTool("adjust")) return;

        this.imageAdjustmentView = new ImageAdjustmentView(this.config, this.imageView.src, this.node.element.parentElement),
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
            this.imageView.setImage(this.imageAdjustmentView.getImageSource());
            await this.waitForLoad();
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
        if (!isEmpty(this.imageDownscaleWindow)) {
            if (intendedAction !== "downscale") {
                this.editor.application.notifications.push(
                    "warn",
                    "Finish Downscaling",
                    `Complete your downscale selection or cancel before trying to ${intendedAction}.`
                );
            } else {
                this.imageDownscaleWindow.focus();
            }
            return true;
        }
        return false;
    }

    /**
     * Opens the image in a new window
     */
    async sendToWindow() {
        const url = URL.createObjectURL(await this.getBlob());
        window.open(url, "_blank");
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
            release();
        }, this.constructor.hideTime);
    }

    /**
     * The callback for MouseEnter
     */
    async onMouseEnter(e) {
        this.stopHideTimer();
        let release = await this.lock.acquire();
        release();
    }

    /**
     * The callback for MouesLeave
     */
    async onMouseLeave(e) {
        this.startHideTimer();
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
     * Gets frame time in milliseconds
     */
    get frameTime() {
       return 1000.0 / this.playbackRate;
    }

    /**
     * Gets sample IDs mapped to images
     */
    get sampleUrls() {
        return isEmpty(this.samples)
            ? []
            : this.isIntermediate
                ? this.samples.map((v) => `${this.model.api.baseUrl}/invocation/intermediates/${v}.png`)
                : this.samples.map((v) => `${this.model.api.baseUrl}/invocation/images/${v}.png`);
    }

    /**
     * Gets sample IDs mapped to thumbnails
     */
    get thumbnailUrls() {
        return isEmpty(this.samples)
            ? []
            : this.isIntermediate
                ? this.samples.map((v) => `${this.model.api.baseUrl}/invocation/intermediates/${v}.png`)
                : this.samples.map((v) => `${this.model.api.baseUrl}/invocation/thumbnails/${v}.png`);
    }

    /**
     * Sets samples
     */
    setSamples(sampleImages, isAnimation) {
        // Get IDs from samples
        if (isEmpty(sampleImages)) {
            this.samples = null;
            this.images.removeClass("has-sample");
        } else {
            this.samples = sampleImages.map((v) => v.split("/").slice(-1)[0].split(".")[0]);
            if (!isEmpty(this.activeIndex)) {
                this.images.addClass("has-sample");
            }
        }

        this.isIntermediate = !isEmpty(this.samples) && sampleImages[0].indexOf("intermediate") !== -1;
        this.isAnimation = isAnimation;

        this.sampleChooser.setIsAnimation(isAnimation);
        this.sampleChooser.setSamples(this.thumbnailUrls).then(() => {
            this.sampleChooser.setActiveIndex(this.activeIndex, false);
        });

        this.sampleViewer.setImage(isAnimation ? this.sampleUrls : isEmpty(this.activeIndex) ? null : this.sampleUrls[this.activeIndex]);
        if (this.isAnimation) {
            this.sampleViewer.setFrame(this.activeIndex);
        }
        if (!isEmpty(this.activeIndex)) {
            waitFor(() => !isEmpty(this.sampleViewer.width) && !isEmpty(this.sampleViewer.height)).then(() => {
                this.images.setDimension(this.sampleViewer.width, this.sampleViewer.height);
            });
        }
    }

    /**
     * Gets active image
     */
    getActiveImage() {
        
    }

    /**
     * Sets the active index when looking at images
     */
    setActive(activeIndex) {
        this.activeIndex = activeIndex;
        if (this.isAnimation) {
            this.sampleChooser.setActiveIndex(activeIndex, false);
            this.sampleViewer.setFrame(activeIndex);
        } else {
            this.sampleViewer.setImage(this.sampleUrls[this.activeIndex]);
        }

        if (isEmpty(activeIndex)) {
            this.images.removeClass("has-sample");
            this.images.setDimension(this.engine.width, this.engine.height);
            this.sampleViewer.hide();
        } else {
            waitFor(() => !isEmpty(this.sampleViewer.width) && !isEmpty(this.sampleViewer.height)).then(() => {
                this.images.setDimension(this.sampleViewer.width, this.sampleViewer.height);
                this.images.addClass("has-sample");
            });
        }
    }

    /**
     * Ticks the animation to the next frame
     */
    tickAnimation() {
        if (isEmpty(this.samples)) return;
        let frameStart = (new Date()).getTime();
        requestAnimationFrame(() => {
            let activeIndex = this.activeIndex,
                frameLength = this.samples.length,
                nextIndex = activeIndex + 1;

            if (this.isPlaying) {
                if (nextIndex < frameLength) {
                    this.setActive(nextIndex);
                } else if(this.isLooping) {
                    this.setActive(0);
                } else {
                    this.sampleChooser.setPlayAnimation(false);
                    return;
                }
                let frameTime = (new Date()).getTime() - frameStart;
                clearTimeout(this.tick);
                this.tick = setTimeout(
                    () => this.tickAnimation(),
                    this.frameTime - frameTime
                );
            }
        });
    }

    /**
     * Modifies playback rate
     */
    async setPlaybackRate(playbackRate, updateChooser = true) {
        this.playbackRate = playbackRate;
        if (updateChooser) {
            this.sampleChooser.setPlaybackRate(playbackRate);
        }
    }

    /**
     * Starts/stops playing
     */
    async setPlay(playing, updateChooser = true) {
        this.isPlaying = playing;
        if (playing) {
            if (this.activeIndex >= this.samples.length - 1) {
                // Reset animation
                this.setActive(0);
            }
            clearTimeout(this.tick);
            this.tick = setTimeout(
                () => this.tickAnimation(),
                this.frameTime
            );
        } else {
            clearTimeout(this.tick);
        }
        if (updateChooser) {
            this.sampleChooser.setPlayAnimation(playing);
        }
    }

    /**
     * Enables/disables looping
     */
    async setLoop(loop, updateChooser = true) {
        this.isLooping = loop;
        if (updateChooser) {
            this.sampleChooser.setLoopAnimation(loop);
        }
    }

    /**
     * Sets horizontal tiling
     */
    async setTileHorizontal(tile, updateChooser = true) {
        this.tileHorizontal = tile;
        this.sampleViewer.tileHorizontal = tile;
        requestAnimationFrame(() => {
            if (updateChooser) {
                this.sampleChooser.setHorizontalTile(tile);
            }
            this.sampleViewer.checkVisibility();
        });
    }

    /**
     * Sets vertical tiling
     */
    async setTileVertical(tile, updateChooser = true) {
        this.tileVertical = tile;
        this.sampleViewer.tileVertical = tile;
        requestAnimationFrame(() => {
            if (updateChooser) {
                this.sampleChooser.setVerticalTile(tile);
            }
            this.sampleViewer.checkVisibility();
        });
    }

    /**
     * Shows the canvas, hiding samples
     */
    async showCanvas(updateChooser = true) {
        this.setPlay(false);
        this.sampleViewer.hide();
        if (updateChooser) {
            this.sampleChooser.setActiveIndex(null);
        }
        this.images.setDimension(this.engine.width, this.engine.height);
    }

    /**
     * On initialize, add DOM nodes
     */
    async initialize() {
        // Create views
        this.sampleChooser = new SampleChooserView(this.config);
        this.sampleViewer = new SampleView(this.config);

        // Bind chooser events
        this.sampleChooser.onShowCanvas(() => this.showCanvas(false));
        this.sampleChooser.onLoopAnimation((loop) => this.setLoop(loop, false));
        this.sampleChooser.onPlayAnimation((play) => this.setPlay(play, false));
        this.sampleChooser.onTileHorizontal((tile) => this.setTileHorizontal(tile, false));
        this.sampleChooser.onTileVertical((tile) => this.setTileVertical(tile, false));
        this.sampleChooser.onSetActive((active) => this.setActive(active, false));
        this.sampleChooser.onSetPlaybackRate((rate) => this.setPlaybackRate(rate, false));

        // Get initial variables
        this.activeIndex = 0;
        this.playbackRate = SampleChooserView.playbackRate;

        // Add chooser to main container
        this.application.container.appendChild(await this.sampleChooser.render());

        // Get image editor in DOM
        let imageEditor = await this.images.getNode();

        // Add sample viewer to canvas
        imageEditor.find("enfugue-node-canvas").append(await this.sampleViewer.getNode());
    }

    /**
     * Gets default state, no samples
     */
    getDefaultState() {
        return {
            "samples": {
                "urls": null,
                "active": null,
                "animation": false
            }
        };
    }

    /**
     * Get state is only for UI; only use the sample choosers here
     */
    getState(includeImages = true) {
        if (!includeImages) {
            return this.getDefaultState();
        }

        return {
            "samples": {
                "urls": this.sampleUrls,
                "active": this.activeIndex,
                "animation": this.isAnimation
            }
        };
    }

    /**
     * Set state is only for UI; set the sample choosers here
     */
    setState(newState) {
        if (isEmpty(newState) || isEmpty(newState.samples) || isEmpty(newState.samples.urls)) {
            this.setSamples(null);
        } else {
            this.activeIndex = newState.samples.active;
            this.setSamples(
                newState.samples.urls,
                newState.samples.animation === true
            );
        }
    }
}

export { SamplesController };
