/** @module view/nodes/image-editor/filter.mjs */
import { isEmpty } from "../../../base/helpers.mjs";
import { ElementBuilder } from "../../../base/builder.mjs";
import { View } from "../../base.mjs";
import { ImageAdjustmentFilter } from "../../../graphics/image-adjust.mjs";
import { ImagePixelizeFilter } from "../../../graphics/image-pixelize.mjs";
import { ImageGaussianBlurFilter } from "../../../graphics/image-blur.mjs";
import { FormView } from "../../forms/base.mjs";
import {
    SliderPreciseInputView,
    SelectInputView
} from "../../forms/input.mjs";

const E = new ElementBuilder();

/**
 * Creates a view for selecting filters
 */
class FilterSelectInputView extends SelectInputView {
    /**
     * @var object The options for this input
     */
    static defaultOptions = {
        "pixelize": "Pixelize",
        "gaussian": "Gaussian Blur"
    };
}

/**
 * Creates a common form view base for filter forms
 */
class ImageFilterFormView extends FormView {
    /**
     * @var bool autosubmit
     */
    static autoSubmit = true;

    /**
     * @var bool Disable disabling
     */
    static disableOnSubmit = false;

    /**
     * Fieldsets include the main filter, then inputs for filter types
     */
    static fieldSets = {
        "Filter": {
            "filter": {
                "class": FilterSelectInputView,
                "config": {
                    "required": true
                }
            }
        },
        "Pixelize": {
            "size": {
                "label": "Pixel Size",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "value": 1
                }
            }
        },
        "Blur": {
            "radius": {
                "label": "Blur Radius",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "value": 1
                }
            }
        }
    };

    /**
     * @var object Default values
     */
    static defaultValues = {
        "size": 10
    };

    /**
     * @var object Callable conditions for fieldset display
     */
    static fieldSetConditions = {
        "Pixelize": (values) => values.filter === "pixelize",
        "Blur": (values) => values.filter === "gaussian"
    };
};

/**
 * Creates a form view for controlling the ImageAdjustmentFilter
 */
class ImageAdjustmentFormView extends ImageFilterFormView {
    /**
     * @var object Various options available
     */
    static fieldSets = {
        "Color Channel Adjustments": {
            "red": {
                "label": "Red Amount",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            },
            "green": {
                "label": "Green Amount",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            },
            "blue": {
                "label": "Blue Amount",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            }
        },
        "Brightness and Contrast": {
            "brightness": {
                "label": "Brightness Adjustment",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            },
            "contrast": {
                "label": "Contrast Adjustment",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            }
        },
        "Hue, Saturation and Lightness": {
            "hue": {
                "label": "Hue Shift",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            },
            "saturation": {
                "label": "Saturation Adjustment",
                "class": SliderPreciseInputView,
                "config": {
                    "min": -100,
                    "max": 100,
                    "value": 0
                }
            },
            "lightness": {
                "label": "Lightness Enhancement",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 0
                }
            }
        },
        "Noise": {
            "hueNoise": {
                "label": "Hue Noise",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 0
                }
            },
            "saturationNoise": {
                "label": "Saturation Noise",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 0
                }
            },
            "lightnessNoise": {
                "label": "Lightness Noise",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 0
                }
            }
        }
    };

    /**
     * @var object Default values
     */
    static defaultValues = {
        "red": 0,
        "green": 0,
        "blue": 0,
        "brightness": 0,
        "contrast": 0,
        "hue": 0,
        "saturation": 0,
        "lightness": 0,
        "hueNoise": 0,
        "saturationNoise": 0,
        "lightnessNoise": 0
    };
};

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
    createFilter(filterType) {
        switch (filterType) {
            case "gaussian":
                return new ImageGaussianBlurFilter(this.image);
            case "pixelize":
                return new ImagePixelizeFilter(this.image);
            case "adjust":
                return new ImageAdjustmentFilter(this.image);
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
        if (!isEmpty(values.filter)) {
            if (this.filterType !== values.filter) {
                // Filter changed
                this.removeCanvas();
                this.filter = this.createFilter(values.filter);
                this.filterType = values.filter;
                this.filter.getCanvas().then((canvas) => {
                    this.canvas = canvas;
                    this.container.appendChild(this.canvas);
                });
            }
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
    ImageFilterFormView,
    ImageAdjustmentFormView,
    ImageFilterView,
    ImageAdjustmentView
};
