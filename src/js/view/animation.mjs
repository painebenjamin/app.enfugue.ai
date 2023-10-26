/** @module view/animation */
import { isEmpty, waitFor } from "../base/helpers.mjs";
import { View } from "./base.mjs";
import { ImageView } from "./image.mjs";

/**
 * The AnimationView extends the ImageView for working with animations.
 */
class AnimationView extends View {
    /**
     * @var string The tag name
     */
    static tagName = "enfugue-animation-view";

    /**
     * On construct, check if we're initializing with sources
     */
    constructor(config, images = []){
        super(config);
        this.canvas = document.createElement("canvas");
        this.loadedCallbacks = [];
        this.setImages(images);
    }

    /**
     * Adds a callback to fire when imgaes are loaded
     */
    onLoad(callback) {
        if (this.loaded) {
            callback(this);
        } else {
            this.loadedCallbacks.push(callback);
        }
    }

    /**
     * On set image, wait for load then trigger callbacks
     */
    setImages(images) {
        this.images = images;
        if (isEmpty(images)) {
            this.loaded = true;
            this.clearCanvas();
        } else {
            this.loaded = false;
            this.imageViews = images.map(
                (image) => new ImageView(this.config, image, false)
            );
            Promise.all(
                this.imageViews.map(
                    (imageView) => imageView.waitForLoad()
                )
            ).then(() => this.imagesLoaded());
        }
    }

    /**
     * When images are loaded, fire callbacks
     */
    async imagesLoaded() {
        this.loaded = true;

        if (!isEmpty(this.imageViews)) {
            this.width = this.imageViews[0].width;
            this.height = this.imageViews[0].height;

            this.canvas.width = this.width;
            this.canvas.height = this.height;

            let context = this.canvas.getContext("2d");
            context.drawImage(this.imageViews[0].image, 0, 0);

            if (this.node !== undefined) {
                this.node.css({
                    "width": this.width,
                    "height": this.height
                });
            }
        }

        for (let callback of this.loadedCallbacks) {
            await callback();
        }
    }

    /**
     * Waits for the promise boolean to be set
     */
    waitForLoad() {
        return waitFor(() => this.loaded);
    }

    /**
     * Sets the frame index
     */
    setFrame(index) {
        if (isEmpty(index)) index = 0;
        this.frame = index;
        if (this.loaded) {
            let context = this.canvas.getContext("2d");
            context.drawImage(this.imageViews[this.frame].image, 0, 0);
        } else {
            this.waitForLoad().then(() => this.setFrame(index));
        }
    }

    /**
     * Clears the canvas
     */
    clearCanvas() {
        let context = this.canvas.getContext("2d");
        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    /**
     * On build, append canvas
     */
    async build() {
        let node = await super.build();
        node.content(this.canvas);
        if (this.loaded) {
            node.css({
                "width": this.width,
                "height": this.height
            });
        }
        return node;
    }
}

export { AnimationView };
