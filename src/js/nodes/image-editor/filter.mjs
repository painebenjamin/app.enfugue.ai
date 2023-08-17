/** @module nodes/image-editor/filter.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { View } from "../../view/base.mjs";
import { ImageAdjustmentFilter } from "../../graphics/image-adjust.mjs";
import { ImagePixelizeFilter } from "../../graphics/image-pixelize.mjs";
import { ImageSharpenFilter } from "../../graphics/image-sharpen.mjs";
import {
    ImageBoxBlurFilter,
    ImageGaussianBlurFilter
} from "../../graphics/image-blur.mjs";
import {
    ImageFilterFormView,
    ImageAdjustmentFormView
} from "../../forms/enfugue/image-editor.mjs";

const E = new ElementBuilder();

/**
 * Combines the a filter form view and various buttons for executing
 */
class ImageFilterView extends View {
    /**
     * @var class The class of the filter form.
     */
    static filterFormView = ImageFilterFormView;

    /**
     * On construct, build form and bind submit
     */
    constructor(config, image, container) {
        super(config);
        this.image = image;
        this.container = container;
        this.cancelCallbacks = [];
        this.saveCallbacks = [];
        this.formView = new this.constructor.filterFormView(config);
        this.formView.onSubmit((values) => {
            this.setFilter(values);
        });
    }

    /**
     * Creates a GPU-accelerated filter helper using the image
     */
    createFilter(filterType, execute = true) {
        switch (filterType) {
            case "box":
                return new ImageBoxBlurFilter(this.image, execute);
            case "gaussian":
                return new ImageGaussianBlurFilter(this.image, execute);
            case "sharpen":
                return new ImageSharpenFilter(this.image, execute);
            case "pixelize":
                return new ImagePixelizeFilter(this.image, execute);
            case "adjust":
                return new ImageAdjustmentFilter(this.image, execute);
            case "invert":
               return new ImageAdjustmentFilter(this.image, execute, {invert: 1});
            default:
                this.editor.application.notifications.push("error", `Unknown filter ${filterType}`);
        }
    }

    /**
     * Gets the image source from the filter, if present
     */
    getImageSource() {
        if (!isEmpty(this.filter)) {
            return this.filter.imageSource;
        }
        return this.image;
    }

    /**
     * Sets the filter and filter constants
     */
    setFilter(values) {
        if (isEmpty(values.filter)) {
            this.removeCanvas();
        } else if (this.filterType !== values.filter) {
            // Filter changed
            this.removeCanvas();
            this.filter = this.createFilter(values.filter, false);
            this.filterType = values.filter;
            this.filter.getCanvas().then((canvas) => {
                this.filter.setConstants(values);
                this.canvas = canvas;
                this.container.appendChild(this.canvas);
i            });
        }

        if (!isEmpty(this.filter)) {
            this.filter.setConstants(values);
        }
    }

    /**
     * Removes the canvas if its attached
     */
    removeCanvas() {
        if (!isEmpty(this.canvas)) {
            try {
                this.container.removeChild(this.canvas);
            } catch(e) { }
            this.canvas = null;
        }
    }

    /**
     * @param callable $callback Method to call when 'cancel' is clicked
     */
    onCancel(callback) {
        this.cancelCallbacks.push(callback);
    }

    /**
     * @param callable $callback Method to call when 'save' is clicked
     */
    onSave(callback) {
        this.saveCallbacks.push(callback);
    }

    /**
     * Call all save callbacks
     */
    async saved() {
        for (let saveCallback of this.saveCallbacks) {
            await saveCallback();
        }
    }
    
    /**
     * Call all cancel callbacks
     */
    async canceled() {
        for (let cancelCallback of this.cancelCallbacks) {
            await cancelCallback();
        }
    }

    /**
     * On build, add buttons and bind callbacks
     */
    async build() {
        let node = await super.build(),
            reset = E.button().class("column").content("Reset"),
            save = E.button().class("column").content("Save"),
            cancel = E.button().class("column").content("Cancel"),
            nodeButtons = E.div().class("flex-columns half-spaced margin-top padded-horizontal").content(
                reset,
                save,
                cancel
            );

        reset.on("click", () => {
            this.formView.setValues(this.constructor.filterFormView.defaultValues);
            setTimeout(() => { this.formView.submit(); }, 100);
        });
        save.on("click", () => this.saved());
        cancel.on("click", () => this.canceled());
        node.content(
            await this.formView.getNode(),
            nodeButtons
        );
        return node;
    }
};

/**
 * Combines the adjustment form view and application buttons
 */
class ImageAdjustmentView extends ImageFilterView {
    /**
     * @var class The class of the filter form.
     */
    static filterFormView = ImageAdjustmentFormView;
    
    /**
     * On construct, build form and bind submit
     */
    constructor(config, image, container) {
        super(config, image, container);
        this.setFilter({"filter": "adjust"});
    }
}

export {
    ImageFilterView,
    ImageAdjustmentView
};
