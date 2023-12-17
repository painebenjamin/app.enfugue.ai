/** @module forms/input/enfugue/engine */
import { isEmpty, deepClone, createElementsFromString } from "../../../base/helpers.mjs";
import {
    NumberInputView,
    FloatInputView,
    SliderPreciseInputView,
} from "../numeric.mjs";
import {
    FormInputView,
    RepeatableInputView
} from "../parent.mjs";
import { FormView } from "../../base.mjs";
import { CheckboxInputView } from "../bool.mjs";
import { SelectInputView } from "../enumerable.mjs";

/**
 * Engine size input
 */
class EngineSizeInputView extends NumberInputView {
    /**
     * @var int Minimum pixel size
     */
    static min = 128;

    /**
     * @var int Maximum pixel size
     */
    static max = 2048;

    /**
     * @var int Multiples of 8
     */
    static step = 8;

    /**
     * @var string The tooltip to display to the user
     */
    static tooltip = "When using tiled diffusion, this is the size of the window (in pixels) that will be encoded, decoded or inferred at once. When left blank, the tile size is equal to the training size of the base model - 512 for Stable Diffusion 1.5, or 1024 for Stable Diffusion XL.";
};

/**
 * Default VAE Input View
 */
class DefaultVaeInputView extends SelectInputView {
    /**
     * @var object Option values and labels
     */
    static defaultOptions = {
        "ema": "EMA 560000",
        "mse": "MSE 840000",
        "xl": "SDXL",
        "xl16": "SDXL FP16",
        "other": "Other"
    };
    
    /**
     * @var string Default text
     */
    static placeholder = "Default";

    /**
     * @var bool Allow null
     */
    static allowEmpty = true;

    /**
     * @var string Tooltip to display
     */
    static tooltip = "Variational Autoencoders are the model that translates images between pixel space - images that you can see - and latent space - images that the AI model understands. In general you do not need to select a particular VAE model, but you may find slight differences in sharpness of resulting images.";
};

/**
 * Mask Type Input View
 */
class MaskTypeInputView extends SelectInputView {
    /**
     * @var object Option values and labels
     */
    static defaultOptions = {
        "constant": "Constant",
        "bilinear": "Bilinear",
        "gaussian": "Gaussian"
    };

    /**
     * @var string The tooltip
     */
    static tooltip = "During multi-diffusion (tiled diffusion), only a square of the size of the engine is rendered at any given time. This can cause hard edges between the frames, especially when using a large stride. Using a mask allows for blending along the edges - this can remove seams, but also reduce precision.";

    /**
     * @var string Default value
     */
    static defaultValue = "bilinear";
}

/**
 * Scheduler Input View
 */
class SchedulerInputView extends SelectInputView {
    /**
     * @var object Option values and labels
     */
    static defaultOptions = {
        "ddim": "DDIM: Denoising Diffusion Implicit Models",
        "ddpm": "DDPM: Denoising Diffusion Probabilistic Models",
        "deis": "DEIS: Diffusion Exponential Integrator Sampler",
        "dpmss": "DPM-Solver++ SDE",
        "dpmssk": "DPM-Solver++ SDE Karras",
        "dpmsm": "DPM-Solver++ 2M",
        "dpmsmk": "DPM-Solver++ 2M Karras",
        "dpmsms": "DPM-Solver++ 2M SDE",
        "dpmsmka": "DPM-Solver++ 2M SDE Karras",
        "heun": "Heun Discrete Scheduler",
        "dpmd": "DPM Discrete Scheduler (KDPM2)",
        "dpmdk": "DPM Discrete Scheduler (KDPM2) Karras",
        "adpmd": "DPM Ancestral Discrete Scheduler (KDPM2A)",
        "adpmdk": "DPM Ancestral Discrete Scheduler (KDPM2A) Karras",
        "dpmsde": "DPM Solver SDE Scheduler",
        "unipc": "UniPC: Predictor (UniP) and Corrector (UniC)",
        "lmsd": "LMS: Linear Multi-Step Discrete Scheduler",
        "lmsdk": "LMS: Linear Multi-Step Discrete Scheduler Karras",
        "pndm": "PNDM: Pseudo Numerical Methods for Diffusion Models",
        "eds": "Euler Discrete Scheduler",
        "eads": "Euler Ancestral Discrete Scheduler",
        "lcm": "LCM Scheduler"
    };

    /**
     * @var string The tooltip
     */
    static tooltip = "Schedulers control how an image is denoiser over the course of the inference steps. Schedulers can have small effects, such as creating 'sharper' or 'softer' images, or drastically change the way images are constructed. Experimentation is encouraged, if additional information is sought, search <strong>Diffusers Schedulers</strong> in your search engine of choice.";
    
    /**
     * @var string Default text
     */
    static placeholder = "Default";

    /**
     * @var bool Allow null
     */
    static allowEmpty = true;
};

/**
 * Add text for inpainter engine size
 */
class InpainterEngineSizeInputView extends EngineSizeInputView {
    /**
     * @var string The tooltip to display to the user
     */
    static tooltip = "This engine size functions the same as the base engine size, but only applies when inpainting.\n\n" + EngineSizeInputView.tooltip;

    /**
     * @var ?int no default value
     */
    static defaultValue = null;
};

/**
 * Add text for refiner engine size
 */
class RefinerEngineSizeInputView extends EngineSizeInputView {
    /**
     * @var string The tooltip to display to the user
     */
    static tooltip = "This engine size functions the same as the base engine size, but only applies when refining.\n\n" + EngineSizeInputView.tooltip;

    /**
     * @var ?int no default value
     */
    static defaultValue = null;
};

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
 * Which controlnets are available
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
        "qr": "QR Code",
        "sparse-rgb": "Sparse RGB",
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
        "<strong>QR Code</strong> is a specialized control network designed to generate images from QR codes that are scannable QR codes themselves.<br />" + 
        "<strong>Sparse RGB</strong> is a ControlNet designed for generating videos given one or more images as frames along a timeline. However it can also be used for image generation as a general-purpose reference ControlNet.";

};

/**
 * All options for a control image in a form
 */
class ControlNetUnitFormView extends FormView {
    /**
     * @var object field sets for a control image
     */
    static fieldSets = {
        "Control Unit": {
            "controlnet": {
                "label": "ControlNet",
                "class": ControlNetInputView
            },
            "conditioningScale": {
                "label": "Conditioning Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "How closely to follow ControlNet's influence. Typical values vary, usually values between 0.5 and 1.0 produce good conditioning with balanced randomness, but other values may produce something closer to the desired result."
                }
            },
            "conditioningStart": {
                "label": "Conditioning Start",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.0,
                    "tooltip": "When to begin using this ControlNet for influence. Defaults to the beginning of generation."
                }
            },
            "conditioningEnd": {
                "label": "Conditioning End",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "When to stop using this ControlNet for influence. Defaults to the end of generation."
                }
            },
            "processControlImage": {
                "label": "Process Image for ControlNet",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "When checked, the image will be processed through the appropriate edge detection algorithm for the ControlNet. Only uncheck this if your image has already been processed through edge detection."
                }
            }
        }
    };

    /**
     * @var bool Hide submit button
     */
    static autoSubmit = true;
};

/**
 * The control image form as an input
 */
class ControlNetUnitFormInputView extends FormInputView {
    /**
     * @var class The form class
     */
    static formClass = ControlNetUnitFormView;
};

/**
 * The control image form input as a repeatable input
 */
class ControlNetUnitsInputView extends RepeatableInputView {
    /**
     * @var class the input class
     */
    static memberClass = ControlNetUnitFormInputView;

    /**
     * @var string The text to show when no items present
     */
    static noItemsLabel = "No ControlNet Units";

    /**
     * @var string Text in the add button
     */
    static addItemLabel = "Add ControlNet Unit";
}

/**
 * The beta schedule options
 */
class BetaScheduleInputView extends SelectInputView {
    /**
     * @var bool Allow empty result
     */
    static allowEmpty = true;

    /**
     * @var string Placeholder
     */
    static placeholder = "Default";

    /**
     * @var object Options
     */
    static defaultOptions = {
        "linear": "Linear",
        "scaled_linear": "Scaled Linear",
        "squaredcos_cap_v2": "Squared Cosine"
    };
};

export {
    EngineSizeInputView,
    RefinerEngineSizeInputView,
    InpainterEngineSizeInputView,
    SchedulerInputView,
    MaskTypeInputView,
    ControlNetInputView,
    ControlNetUnitsInputView,
    ImageColorSpaceInputView,
    BetaScheduleInputView,
};
