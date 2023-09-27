/** @module forms/input/enfugue/image-editor */
import { SelectInputView } from "../enumerable.mjs";

/**
 * This input allows the user to specify what colors an image is, so we can determine
 * if we need to invert them on the backend.
 */
class ImageColorSpaceInputView extends SelectInputView {
    /**
     * @var object Only one option
     */
    static defaultOptions = {
        "invert": "White on Black"
    };

    /**
     * @var string The default option is to invert
     */
    static defaultValue = "invert";

    /**
     * @var string The empty option text
     */
    static placeholder = "Black on White";

    /**
     * @var bool Always show empty
     */
    static allowEmpty = true;
}

/**
 * These are the ControlNet options
 */
class ControlNetInputView extends SelectInputView {
    /**
     * @var string Set the default to the easiest and fastest
     */
    static defaultValue = "canny";

    /**
     * @var object The options allowed.
     */
    static defaultOptions = {
        "canny": "Canny Edge Detection",
        "hed": "Holistically-nested Edge Detection (HED)",
        "pidi": "Soft Edge Detection (PIDI)",
        "mlsd": "Mobile Line Segment Detection (MLSD)",
        "line": "Line Art",
        "anime": "Anime Line Art",
        "scribble": "Scribble",
        "depth": "Depth Detection (MiDaS)",
        "normal": "Normal Detection (Estimate)",
        "pose": "Pose Detection (DWPose/OpenPose)",
        "qr": "QR Code"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "The ControlNet to use depends on your input image. Unless otherwise specified, your input image will be processed through the appropriate algorithm for this ControlNet prior to diffusion.<br />" +
        "<strong>Canny Edge</strong>: This network is trained on images and the edges of that image after having run through Canny Edge detection.<br />" +
        "<strong>HED</strong>: Short for Holistically-Nested Edge Detection, this edge-detection algorithm is best used when the input image is too blurry or too noisy for Canny Edge detection.<br />" +
        "<strong>Soft Edge Detection</strong>: Using a Pixel Difference Network, this edge-detection algorithm can be used in a wide array of applications.<br />" +
        "<strong>MLSD</strong>: Short for Mobile Line Segment Detection, this edge-detection algorithm searches only for straight lines, and is best used for geometric or architectural images.<br />" +
        "<strong>Line Art</strong>: This model is capable of rendering images to line art drawings. The controlnet was trained on the model output, this provides a great way to provide your own hand-drawn pieces as well as another means of edge detection.<br />" +
        "<strong>Anime Line Art</strong>: This is similar to the above, but focusing specifically on anime style.<br />" +
        "<strong>Scribble</strong>: This ControlNet was trained on a variant of the HED edge-detection algorithm, and is good for hand-drawn scribbles with thick, variable lines.<br />" +
        "<strong>Depth</strong>: This uses Intel's MiDaS model to estimate monocular depth from a single image. This uses a greyscale image showing the distance from the camera to any given object.<br />" +
        "<strong>Normal</strong>: Normal maps are similar to depth maps, but instead of using a greyscale depth, three sets of distance data is encoded into red, green and blue channels.<br />" +
        "<strong>DWPose/OpenPose</strong>: OpenPose is an AI model from the Carnegie Mellon University's Perceptual Computing Lab detects human limb, face and digit poses from an image, and DWPose is a faster and more accurate model built on top of OpenPose. Using this data, you can generate different people in the same pose.<br />" +
        "<strong>QR Code</strong> is a specialized control network designed to generate images from QR codes that are scannable QR codes themselves.";
};


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
    ControlNetInputView,
    ImageAnchorInputView,
    ImageFitInputView,
    ImageColorSpaceInputView,
    FilterSelectInputView
};
