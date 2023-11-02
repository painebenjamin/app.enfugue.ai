/** @module forms/input/enfugue/image-editor */
import { SelectInputView } from "../enumerable.mjs";

/**
 * The fit options
 */
class ImageFitInputView extends SelectInputView {
    /**
     * @var string The default value is the first
     */
    static defaultValue = "actual";

    /**
     * @var object The Options allowed
     */
    static defaultOptions = {
        "actual": "Actual Size",
        "stretch": "Stretch",
        "contain": "Contain",
        "cover": "Cover"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "Fit this image within it's container.<br /><strong>Actual</strong>: Do not manipulate the dimensions of the image, anchor it as specified in the frame.<br /><strong>Stretch</strong>: Force the image to fit within the frame, regardles sof original dimensions. When using this mode, anchor has no effect.<br /><strong>Contain</strong>: Scale the image so that it's largest dimension is contained within the frames bounds, adding negative space to fill the rest of the frame.<br /><strong>Cover</strong>: Scale the image so that it's smallest dimension is contained within the frame bounds, cropping the rest of the image as needed.";
};

/**
 * The anchor options
 */
class ImageAnchorInputView extends SelectInputView {
    /**
     * @var string The default value is the first
     */
    static defaultValue = "top-left";

    /**
     * @var object The Options allowed
     */
    static defaultOptions = {
        "top-left": "Top Left",
        "top-center": "Top Center",
        "top-right": "Top Right",
        "center-left": "Center Left",
        "center-center": "Center Center",
        "center-right": "Center Right",
        "bottom-left": "Bottom Left",
        "bottom-center": "Bottom Center",
        "bottom-right": "Bottom Right",
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "When the size of the frame and the size of the image do not match, this will control where the image is placed. View the 'fit' field for more options to fit images in the frame.";
};

/**
 * The filters allowed on the image
 */
class FilterSelectInputView extends SelectInputView {
    /**
     * @var string The value of the empty item
     */
    static placeholder = "None";

    /**
     * @var bool Allow none
     */
    static allowEmpty = true;

    /**
     * @var object Options
     */
    static defaultOptions = {
        "invert": "Invert",
        "pixelize": "Pixelize",
        "box": "Box Blur",
        "gaussian": "Gaussian Blur",
        "sharpen": "Sharpen"
    };
};

export {
    ImageAnchorInputView,
    ImageFitInputView,
    FilterSelectInputView
};
