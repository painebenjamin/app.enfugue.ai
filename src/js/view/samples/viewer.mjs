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
        this.imageViews = (new Array(9)).fill(null).map(() => new ImageView(this.config));
        this.image = null;
        this.tileHorizontal = false;
        this.tileVertical = false;
    }

    /**
     * @var string The tag name
     */
    static tagName = "enfugue-sample";

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
     * Sets the image, either a single image or multiple
     */
    setImage(image) {
        this.image = image;
        if (Array.isArray(image)) {
            for (let animationView of this.animationViews) {
                animationView.setImages(image);
                animationView.setFrame(0);
            }
        } else {
            for (let imageView of this.imageViews) {
                imageView.setImage(image);
            }
        }
        window.requestAnimationFrame(() => this.checkVisibility());
    }

    /**
     * Sets the frame for animations
     */
    setFrame(frame) {
        for (let animationView of this.animationViews) {
            animationView.setFrame(frame);
        }
    }

    /**
     * On build, bind mouseenter to show tools
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
