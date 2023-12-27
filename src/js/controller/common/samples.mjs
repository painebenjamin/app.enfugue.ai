/** @module controller/common/samples */
import {
    downloadAsDataURL,
    isEmpty,
    waitFor,
    sleep,
    isEquivalent
} from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { SimpleNotification } from "../../common/notify.mjs";
import { SampleChooserView } from "../../view/samples/chooser.mjs";
import { SampleView } from "../../view/samples/viewer.mjs";
import {
    ImageAdjustmentView,
    ImageFilterView
} from "../../view/samples/filter.mjs";
import { View } from "../../view/base.mjs";
import { ImageView } from "../../view/image.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import {
    UpscaleFormView,
    DownscaleFormView
} from "../../forms/enfugue/upscale.mjs";
import { QuickStableVideoDiffusionFormView } from "../../forms/enfugue/animation.mjs";

const E = new ElementBuilder();

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
    static imageAdjustmentWindowHeight = 450;
    
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
     * @var int The width of the downscale window in pixels
     */
    static imageDownscaleWindowWidth = 260;

    /**
     * @var int The height of the downscale window in pixels
     */
    static imageDownscaleWindowHeight = 210;

    /**
     * @var int The width of the stable video window in pixels
     */
    static stableVideoWindowWidth = 400;

    /**
     * @var int The height of the stable video window in pixels
     */
    static stableVideoWindowHeight = 1000;

    /**
     * Adds the image menu to the passed menu
     */
    async prepareImageMenu(menu) {
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

        let sendToSVD = await menu.addItem("Send to SVD", "fa-solid fa-video", "v");
        sendToSVD.onClick(() => this.sendToSVD());
    }

    /**
     * Adds the video menu to the passed menu
     */
    async prepareVideoMenu(menu) {
        let popoutVideo = await menu.addItem("Popout Video", "fa-solid fa-arrow-up-right-from-square", "p");
        popoutVideo.onClick(() => this.sendVideoToWindow());

        let saveVideo = await menu.addItem("Save As", "fa-solid fa-floppy-disk", "a");
        saveVideo.onClick(() => this.saveVideoToDisk());

        let editVideo = await menu.addItem("Edit Video", "fa-solid fa-pen-to-square", "t");
        editVideo.onClick(() => this.sendVideoToCanvas());

        let upscaleVideo = await menu.addItem("Upscale Video", "fa-solid fa-up-right-and-down-left-from-center", "u");
        upscaleVideo.onClick(() => this.startVideoUpscale());

        let gifVideo = await menu.addItem("Get as GIF", "fa-solid fa-file-video", "g");
        gifVideo.onClick(() => this.getVideoGif());
        /**
        let interpolateVideo = await menu.addItem("Interpolate Video", "fa-solid fa-film", "i");
        interpolateVideo.onClick(() => this.startVideoInterpolate());
        */
    }

    /**
     * Triggers the copy to clipboard
     */
    async copyToClipboard() {
        navigator.clipboard.write([
            new ClipboardItem({
                "image/png": await this.sampleViewer.getBlob()
            })
        ]);
        SimpleNotification.notify("Copied to clipboard!", 2000);
    }

    /**
     * Saves the image to disk
     * Asks for a filename first
     */
    async saveToDisk() {
        this.application.saveBlobAs(
            "Save Image",
            await this.sampleViewer.getBlob(),
            ".png"
        );
    }

    /**
     * Saves the image to disk
     * Asks for a filename first
     */
    async saveVideoToDisk() {
        this.application.saveRemoteAs("Save Video", this.video);
    }

    /**
     * Sends the image to a new canvas
     */
    async sendToCanvas() {
        return await this.application.initializeStateFromImage(
            this.sampleViewer.getDataURL(),
            true, // Save history
            null, // Prompt for settings
            null, // No state overrides
            false, // Not video
        );
    }

    /*
     * Sends the video to a new canvas
     */
    async sendVideoToCanvas() {
        return await this.application.initializeStateFromImage(
            await downloadAsDataURL(this.video),
            true, // Save history
            null, // Prompt for settings
            {"samples": {"video": null}}, // State overrides
            true, // Video
        );
    }

    /**
     * Opens the video as a gif
     */
    async getVideoGif() {
        if (isEmpty(this.video) || !this.video.endsWith("mp4")) {
            throw `Video is empty or not a URL.`;
        }
        let gifURL = this.video.substring(0, this.video.length-4) + ".gif";
        window.open(gifURL, "_blank");
    }

    /**
     * Starts downscaling the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageDownscale() {
        if (this.checkActiveTool("downscale")) return;

        let imageBeforeDownscale = this.sampleViewer.getDataURL(),
            widthBeforeDownscale = this.sampleViewer.width,
            heightBeforeDownscale = this.sampleViewer.height,
            setDownscaleAmount = async (amount) => {
                let image = new ImageView(this.config, imageBeforeDownscale);
                await image.waitForLoad();
                await image.downscale(amount);
                this.sampleViewer.setImage(image.src);
                this.application.images.setDimension(
                    image.width,
                    image.height,
                    false
                );
            },
            saveResults = false;

        this.imageDownscaleForm = new DownscaleFormView(this.config);
        this.imageDownscaleWindow = await this.application.windows.spawnWindow(
            "Downscale Image",
            this.imageDownscaleForm,
            this.constructor.imageDownscaleWindowWidth,
            this.constructor.imageDownscaleWindowHeight
        );
        this.imageDownscaleWindow.onClose(() => {
            this.imageDownscaleForm = null;
            this.imageDownscaleWindow = null;
            if (!saveResults) {
                this.sampleViewer.setImage(imageBeforeDownscale);
                this.application.images.setDimension(widthBeforeDownscale, heightBeforeDownscale, false);
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
        this.imageUpscaleWindow = await this.application.windows.spawnWindow(
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
            console.log(values);
            console.log(this.sampleViewer.width);
            console.log(this.sampleViewer.height);
            await this.application.layers.emptyLayers();
            await this.application.canvas.setDimension(
                this.sampleViewer.width,
                this.sampleViewer.height,
                false,
                true
            );
            await this.application.layers.setState({layers: [{
                "classname": "ImageEditorImageNodeView",
                "x": 0,
                "y": 0,
                "w": this.sampleViewer.width,
                "h": this.sampleViewer.height,
                "src": this.sampleViewer.getDataURL(),
                "visibility": "visible"
            }]});
            this.publish("quickUpscale", values);
            // Remove window
            this.imageUpscaleWindow.remove();
            // Wait a tick then trigger invoke
            setTimeout(() => {
                this.application.publish("tryInvoke");
            }, 1000);
        });
    }

    /**
     * Starts upscaling the video
     * Does not replace the current visible canvas.
     * This will use the canvas and upscale settings to send to the backend.
     */
    async startVideoUpscale() {
        if (this.checkActiveTool("upscale")) return;

        this.videoUpscaleForm = new UpscaleFormView(this.config);
        this.videoUpscaleWindow = await this.application.windows.spawnWindow(
            "Upscale Video",
            this.videoUpscaleForm,
            this.constructor.imageUpscaleWindowWidth,
            this.constructor.imageUpscaleWindowHeight
        );
        this.videoUpscaleWindow.onClose(() => {
            this.videoUpscaleForm = null;
            this.videoUpscaleWindow = null;
        });
        this.videoUpscaleForm.onCancel(() => this.videoUpscaleWindow.remove());
        this.videoUpscaleForm.onSubmit(async (values) => {
            await this.application.canvas.setDimension(
                this.sampleViewer.width,
                this.sampleViewer.height,
                false,
                true
            );
            await this.application.layers.setState({layers: [{
                "classname": "ImageEditorVideoNodeView",
                "x": 0,
                "y": 0,
                "w": this.sampleViewer.width,
                "h": this.sampleViewer.height,
                "src": await downloadAsDataURL(this.video),
                "visibility": "visible"
            }]});
            this.publish("quickUpscale", values);
            // Remove window
            this.videoUpscaleWindow.remove();
            // Wait a tick then trigger invoke
            setTimeout(() => {
                this.application.publish("tryInvoke");
            }, 1000);
        });
    }

    /**
     * Starts filtering the image
     * Replaces the current visible canvas with an in-progress edit.
     */
    async startImageFilter() {
        if (this.checkActiveTool("filter")) return;

        this.imageFilterView = new ImageFilterView(
            this.config,
            this.sampleViewer.getDataURL(),
            this.sampleViewer.node.element.parentElement
        );
        this.imageFilterWindow = await this.application.windows.spawnWindow(
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
            await this.sampleViewer.setImage(this.imageFilterView.getImageSource());
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

        this.imageAdjustmentView = new ImageAdjustmentView(
            this.config,
            this.sampleViewer.getDataURL(),
            this.sampleViewer.node.element.parentElement
        );
        this.imageAdjustmentWindow = await this.application.windows.spawnWindow(
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
            await this.sampleViewer.setImage(this.imageAdjustmentView.getImageSource());
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
                this.notify(
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
                this.notify(
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
                this.notify(
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
                this.notify(
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
        const url = URL.createObjectURL(await this.sampleViewer.getBlob());
        window.open(url, "_blank");
    }

    /**
     * Opens the video in a new window
     */
    async sendVideoToWindow() {
        window.open(this.video, "_blank");
    }

    /**
     * Opens the image in the SVD dialog
     */
    async sendToSVD() {
        let stableVideoForm = new QuickStableVideoDiffusionFormView(this.config, this.sampleViewer.getDataURL()),
            stableVideoWindow = await this.spawnWindow(
                "Stable Video Diffusion",
                stableVideoForm,
                this.constructor.stableVideoWindowWidth,
                this.constructor.stableVideoWindowHeight
            );

        stableVideoForm.onSubmit(async (values) => {
            stableVideoForm.clearError();
            try {
                let payload = {...values};
                payload.image = this.sampleViewer.getDataURL();
                let result = await this.model.post("/invoke/svd", null, null, payload);
                if (isEmpty(result.uuid)) {
                    throw "Response did not contain a result.";
                }
                this.engine.enableStop();
                this.engine.startSample = true;
                this.engine.canvasInvocation(result.uuid, true);
                this.notify("info", "Success", "Invocation queued, it will begin shortly.");
                stableVideoForm.enable();
            } catch(e) {
                stableVideoForm.setError(e);
                stableVideoForm.enable();
            }
        });
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
    }

    /**
     * The callback for MouesLeave
     */
    async onMouseLeave(e) {
        this.startHideTimer();
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
     * Spawns a video player if one doesn't exist
     */
    async spawnVideoPlayer() {
        if (isEmpty(this.videoPlayerWindow)) {
            this.videoPlayerWindow = await this.application.spawnVideoPlayer(this.video);
            this.videoPlayerWindow.onClose(() => { delete this.videoPlayerWindow; });
        } else {
            this.videoPlayerWindow.focus();
        }
    }

    /**
     * Closes the video player if it exists
     */
    closeVideoPlayer() {
        if (!isEmpty(this.videoPlayerWindow)) {
            this.videoPlayerWindow.remove();
        }
    }

    /**
     * Sets a final video
     */
    setVideo(newVideo) {
        if (this.video !== newVideo) {
            this.closeVideoPlayer();
        }
        this.video = newVideo;
        if (!isEmpty(newVideo)) {
            this.videoToolsMenu.show();
            this.spawnVideoPlayer();
        } else {
            this.videoToolsMenu.hide();
        }
    }

    /**
     * Resets the sample chooser only
     */
    resetSampleChooser() {
        this.sampleChooser.setIsAnimation(false);
        this.sampleChooser.setSamples([]);
        this.setPlay(false);
    }

    /**
     * Removes the result menu from the header menu
     */
    async removeResultMenu(skipAcquire = false) {
        let release;
        if (skipAcquire) {
            release = () => {};
        } else {
            release = await this.lock.acquire();
        }
        try {
            if (!isEmpty(this.activeResultMenu)) {
                this.application.menu.removeCategory("Result");
            }
        } finally {
            release();
        }
    }

    /**
     * Re-initializes the result menu in the header menu
     */
    async prepareResultMenu() {
        let release = await this.lock.acquire();
        try {
            let newResultMenu = this.isAnimation ? "animation" : "image";
            if (this.activeResultMenu !== newResultMenu) {
                await this.removeResultMenu(true);
                let resultMenu = await this.application.menu.addCategory("Result");
                if (this.isAnimation) {
                    this.prepareVideoMenu(resultMenu);
                } else {
                    this.prepareImageMenu(resultMenu);
                }
                this.activeResultMenu = newResultMenu;
            }
         } finally {
            release();
         }
    }

    /**
     * Sets samples
     */
    setSamples(sampleImages, isAnimation) {
        // Get IDs from samples
        let samples = isEmpty(sampleImages) ? null : sampleImages.map((v) => v.split("/").slice(-1)[0].split(".")[0]);
        if (isEquivalent(this.samples, samples)) {
            return;
        }
        if (isEmpty(samples)) {
            this.images.removeClass("has-sample");
            this.removeResultMenu();
        }
        this.samples = samples;

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
        if (!isEmpty(this.activeIndex) && !isEmpty(this.samples)) {
            if (this.isAnimation) {
                this.imageToolsMenu.hide();
                if (!isEmpty(this.video)) {
                    this.videoToolsMenu.show();
                }
            } else {
                this.imageToolsMenu.show();
                this.videoToolsMenu.hide();
            }
            this.prepareResultMenu();
            this.application.layout.showSamples();
            sleep(250).then(() => {
                waitFor(() => !isEmpty(this.sampleViewer.width)).then(() => {
                    this.images.setDimension(this.sampleViewer.width, this.sampleViewer.height, false);
                    this.images.addClass("has-sample");
                });
            });
        } else {
            this.application.layout.checkHideSamples();
        }
    }

    /**
     * Sets the active index when looking at images
     */
    setActive(activeIndex, showSamples = true) {
        this.activeIndex = activeIndex;
        if (this.isAnimation) {
            this.sampleChooser.setActiveIndex(activeIndex, false);
            this.sampleViewer.setFrame(activeIndex);
            if (!isEmpty(activeIndex)) {
                this.imageToolsMenu.hide();
                if (!isEmpty(this.video)) {
                    this.videoToolsMenu.show();
                }
            }
        } else {
            this.sampleViewer.setImage(this.sampleUrls[this.activeIndex]);
            if (!isEmpty(activeIndex)) {
                this.imageToolsMenu.show();
                this.videoToolsMenu.hide();
            }
        }

        if (isEmpty(activeIndex)) {
            this.application.layout.checkHideSamples();
            this.images.removeClass("has-sample");
            this.sampleViewer.hide();
            this.removeResultMenu();
        } else {
            if (showSamples) {
                this.application.layout.showSamples();
                this.prepareResultMenu();
            }
            sleep(250).then(() => {
                waitFor(() => !isEmpty(this.sampleViewer.width)).then(() => {
                    this.images.setDimension(this.sampleViewer.width, this.sampleViewer.height, false);
                    this.images.addClass("has-sample");
                });
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
                    this.setActive(nextIndex, false);
                } else if(this.isLooping) {
                    this.setActive(0, false);
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
            this.application.layout.showSamples();
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
    async setTileHorizontal(tile) {
        this.tileHorizontal = tile;
        this.sampleViewer.tileHorizontal = tile;
        requestAnimationFrame(() => {
            this.sampleViewer.checkVisibility();
        });
    }

    /**
     * Sets vertical tiling
     */
    async setTileVertical(tile) {
        this.tileVertical = tile;
        this.sampleViewer.tileVertical = tile;
        requestAnimationFrame(() => {
            this.sampleViewer.checkVisibility();
        });
    }

    /**
     * On initialize, add DOM nodes
     */
    async initialize() {
        // Create views
        this.sampleChooser = new SampleChooserView(this.config);
        this.sampleViewer = new SampleView(this.config);

        // Bind chooser events
        this.sampleChooser.onLoopAnimation((loop) => this.setLoop(loop, false));
        this.sampleChooser.onPlayAnimation((play) => this.setPlay(play, false));
        this.sampleChooser.onSetActive((active) => this.setActive(active));
        this.sampleChooser.onSetPlaybackRate((rate) => this.setPlaybackRate(rate, false));

        // Create toolbars
        this.imageToolsMenu = new ToolbarView(this.config);
        this.prepareImageMenu(this.imageToolsMenu);
        this.videoToolsMenu = new ToolbarView(this.config);
        this.prepareVideoMenu(this.videoToolsMenu);

        // Get initial variables
        this.activeIndex = 0;
        this.playbackRate = SampleChooserView.playbackRate;

        // Add chooser to main container
        this.application.container.appendChild(await this.sampleChooser.render());

        // Get image editor in DOM
        let imageEditor = await this.images.getNode();

        // Add sample viewer and toolbars to sample canvas
        imageEditor.find("enfugue-node-canvas").append(
            await this.sampleViewer.getNode(),
            E.div().class("sample-tools-container").content(
                await this.imageToolsMenu.getNode(),
                await this.videoToolsMenu.getNode(),
            )
        );
        imageEditor.render();
    }

    /**
     * Gets default state, no samples
     */
    getDefaultState() {
        return {
            "samples": {
                "urls": null,
                "active": null,
                "video": null,
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
                "animation": this.isAnimation,
                "video": this.video
            }
        };
    }

    /**
     * Set state is only for UI; set the sample choosers here
     */
    setState(newState) {
        if (isEmpty(newState) || isEmpty(newState.samples) || isEmpty(newState.samples.urls)) {
            this.setSamples(null);
            this.setVideo(null);
        } else {
            this.activeIndex = newState.samples.active;
            this.setSamples(
                newState.samples.urls,
                newState.samples.animation === true
            );
            this.setVideo(newState.samples.video);
        }
    }
}

export { SamplesController };
