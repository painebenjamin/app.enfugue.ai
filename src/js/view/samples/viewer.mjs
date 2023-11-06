/** @module view/samples/viewer  */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { SimpleNotification } from "../../common/notify.mjs";
import { View } from "../../view/base.mjs";
import { ImageView } from "../../view/image.mjs";
import { AnimationView } from "../../view/animation.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import {
    UpscaleFormView,
    DownscaleFormView
} from "../../forms/enfugue/upscale.mjs";
import {
    ImageAdjustmentView,
    ImageFilterView
} from "./filter.mjs";

const E = new ElementBuilder();

/**
 * This view represents the visible image(s) on the canvas
 */
class SampleView extends View {
    /**
     * Constructed by the editor, pass reference so we can call other functions
     */
    constructor(config) {
        super(config);
        this.animationViews = (new Array(9)).fill(null).map(() => new AnimationView(this.config));
        this.imageViews = (new Array(9)).fill(null).map((_, i) => new ImageView(this.config, null, i === 4));
        this.image = null;
        this.tileHorizontal = false;
        this.tileVertical = false;
    }

    /**
     * @var string The tag name
     */
    static tagName = "enfugue-sample";

    /**
     * @return int width of the sample(s)
     */
    get width() {
        if (isEmpty(this.image)) {
            return null;
        }
        if (Array.isArray(this.image)) {
            return this.animationViews[4].width;
        }
        return this.imageViews[4].width;
    }

    /**
     * @return int height of the sample(s)
     */
    get height() {
        if (isEmpty(this.image)) {
            return null;
        }
        if (Array.isArray(this.image)) {
            return this.animationViews[4].height;
        }
        return this.imageViews[4].height;
    }

    /**
     * Checks and shows what should be shown (if anything)
     */
    checkVisibility() {
        for (let i = 0; i < 9; i++) {
            let imageView = this.imageViews[i],
                animationView = this.animationViews[i],
                isVisible = true;

            switch (i) {
                case 0:
                case 2:
                case 6:
                case 8:
                    isVisible = this.tileHorizontal && this.tileVertical;
                    break;
                case 1:
                case 7:
                    isVisible = this.tileVertical;
                    break;
                case 3:
                case 5:
                    isVisible = this.tileHorizontal;
                    break;
            }

            if (!isVisible) {
                imageView.hide();
                animationView.hide();
            } else if (Array.isArray(this.image)) {
                imageView.hide();
                animationView.show();
            } else {
                imageView.show();
                animationView.hide();
            }
        }
    }

    /**
     * Gets the image view as a blob
     */
    async getBlob() {
        return await this.imageViews[4].getBlob();
    }

    /**
     * Gets the image view as a data URL
     */
    getDataURL() {
        return this.imageViews[4].getDataURL()
    }

    /**
     * Sets the image, either a single image or multiple
     */
    setImage(image) {
        this.image = image;
        if (Array.isArray(image)) {
            for (let animationView of this.animationViews) {
                animationView.setImages(image);
                animationView.setFrame(0);
            }
            window.requestAnimationFrame(() => { 
                this.checkVisibility();
                window.requestAnimationFrame(() => {
                    this.show();
                });
            });
        } else if (!isEmpty(this.image)) {
            for (let imageView of this.imageViews) {
                imageView.setImage(image);
            }
            Promise.all(this.imageViews.map((v) => v.waitForLoad())).then(() => {
                this.checkVisibility();
                window.requestAnimationFrame(() => {
                    this.show();
                });
            });
        } else {
            this.hide();
            for (let animationView of this.animationViews) {
                animationView.clearCanvas();
            }
        }
    }

    /**
     * Sets the frame for animations
     */
    setFrame(frame) {
        this.show();
        for (let animationView of this.animationViews) {
            animationView.setFrame(frame);
        }
    }

    /**
     * On build, add image and animation containers
     */
    async build() {
        let node = await super.build(),
            imageContainer = E.div().class("images-container"),
            animationContainer = E.div().class("animation-container");

        for (let imageView of this.imageViews) {
            imageView.hide();
            imageContainer.append(await imageView.getNode());
        }

        for (let animationView of this.animationViews) {
            animationView.hide();
            animationContainer.append(await animationView.getNode());
        }

        node.content(imageContainer, animationContainer);
        return node;
    }
};

export { SampleView };
