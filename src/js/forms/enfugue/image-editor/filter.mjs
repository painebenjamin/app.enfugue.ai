/** @module forms/enfugue/image-editor/filter */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../../forms/base.mjs";
import {
    FilterSelectInputView,
    SliderPreciseInputView
} from "../../../forms/input.mjs";

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
            }
        },
        "Size": {
            "size": {
                "class": SliderPreciseInputView,
                "config": {
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "value": 4
                }
            }
        },
        "Radius": {
            "radius": {
                "class": SliderPreciseInputView,
                "config": {
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "value": 1
                }
            }
        },
        "Weight": {
            "weight": {
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "value": 0
                }
            }
        }
    };

    /**
     * @var object Default values
     */
    static defaultValues = {
        "filter": null,
        "size": 16,
        "radius": 2,
        "weight": 0
    };

    /**
     * @var object Callable conditions for fieldset display
     */
    static fieldSetConditions = {
        "Size": (values) => ["pixelize"].indexOf(values.filter) !== -1,
        "Radius": (values) => ["gaussian", "box", "sharpen"].indexOf(values.filter) !== -1,
        "Weight": (values) => ["sharpen"].indexOf(values.filter) !== -1
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

export {
    ImageFilterFormView,
    ImageAdjustmentFormView
};
