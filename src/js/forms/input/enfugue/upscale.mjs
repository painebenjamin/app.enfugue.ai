/** @module forms/input/enfugue/upscale */
import { SelectInputView } from "../enumerable.mjs";
import { NumberInputView, FloatInputView, SliderPreciseInputView } from "../numeric.mjs";
import { PromptInputView } from "./prompts.mjs";

/**
 * The input box for outscale. Allows any number between 1.0 and 16.
 */
class UpscaleAmountInputView extends FloatInputView {
    /**
     * @var float The minimum value.
     */
    static min = 0.5;

    /**
     * @var float The default value.
     */
    static defaultValue = 2.0;

    /**
     * @var float The step value
     */
    static step = 0.25;

    /**
     * @var float The maximum scale
     */
    static max = 16.0;
}

/**
 * The upscale mode input view
 * ESRGAN is probably best overall, but we allow a bunch of options
 */
class UpscaleMethodInputView extends SelectInputView {
    /**
     * @var object The option values and label
     */
    static defaultOptions = {
        "esrgan": "ESRGAN",
        "esrganime": "ESRGANime",
        "gfpgan": "GFPGAN",
        "lanczos": "Lanczos",
        "bicubic": "Bicubic",
        "bilinear": "Bilinear",
        "nearest": "Nearest"
    };
    
    /**
     * @var string The default value
     */
    static defaultValue = "esrgan";

    /**
     * @var string The tooltip
     */
    static tooltip = "The upscaling method has a significant effect on the output image. The best general-purpose upscaling method is selected by default.<br />When selecting multiple methods, the first is used for the first upscale, the second for the second (when using iterative upscaling), etc.<br /><strong>ESRGAN</strong>: Short for Enhanced Super-Resolution Generative Adversarial Network, this is an AI upscaling method that tries to maintain sharp edges where they should be sharp, and soft where they should be soft, filling in details along the way.<br /><strong>ESRGANime</strong>: Similar to the above, but with sharper lines for cartoon or anime style.<br /><strong>GFPGAN</strong>: Short for Generative Facial Prior Generative Adversarial Network, this is an AI Upscaling method with face restoration. This results in photorealistic faces more often than not, but can erase desired features; it is best paired with upscale diffusion.<br /><strong>Lanczos</strong>: An algorithm with blurry but consistent results.<br /><strong>Bicubic</strong>: An algorithm that can result in slightly sharper edges than Lanczos, but can have jagged edges on curves and diagonal lines.<br /><strong>Bilinear</strong>: A very fast algorithm with overall the blurriest results.<br /><strong>Nearest</strong>: Maintain sharp pixel boundaries, resulting in a pixelated or retro look.";
}

/**
 * The select box for ControlNet
 * Tile is usually the best for upscaling, but there could be occasions where others
 * give more consistent results
 */
class UpscaleDiffusionControlnetInputView extends SelectInputView {
    /**
     * @var object The option values and label
     */
    static defaultOptions = {
        "tile": "Tile",
        "canny": "Canny Edge Detection",
        "hed": "HED (Holistically-Nested Edge Detection)",
        "pidi": "Soft Edge Detection (PIDI)",
        "depth": "Depth (MiDaS)",
    };
    
    /**
     * @var bool always allow empty
     */
    static allowEmpty = true;

    /**
     * @var string placeholder text
     */
    static placeholder = "None";
    
    /**
     * @var string The tooltip to display
     */
    static tooltip = "The controlnet to use during upscaling. None are required, and using one will result in significant slowdowns during upscaling, but can result in a more consistent upscaled image. When using multiple methods, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.<br /><strong>Tile</strong>: This network is trained on large images and slices of their images.<br /><strong>Canny Edge</strong>: This network is trained on images and the edges of that image after having run through Canny Edge detection. The output image will be processed with this algorithm.<br /><strong>HED</strong>: Short for Holistically-Nested Edge Detection, this edge-detection algorithm is best used when the input image is too blurry or too noisy for Canny Edge detection.<br /><strong>PIDI</strong>: This is an AI edge detection algorithm that can quickly detect edges in a variety of lighting and contrast conditions.<br /><strong>MiDaS</strong>: This algorithm analyzes the image for an approximation of distance from the camera, and can help maintain distance from the camera.";
}

/**
 * The input view for prompt
 */
class UpscaleDiffusionPromptInputView extends PromptInputView {
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The prompt to use when upscaling, it is generally best to use generic detail-oriented prompts, unless there are specific things or people you want to ensure have details.";
}

/**
 * The input view for negative prompt
 */
class UpscaleDiffusionNegativePromptInputView extends PromptInputView {
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The negative prompt to use when upscaling, it is generally best to use generic negative prompts, unless there are specific things you don't want.";
}

/**
 * The input view for strength
 */
class UpscaleDiffusionStrengthInputView extends SliderPreciseInputView {
    /**
     * @var float Min value
     */
    static min = 0.0;

    /**
     * @var float step value
     */
    static step = 0.01;

    /**
     * @var float Max value
     */
    static max = 1.0;

    /**
     * @var float The default value
     */
    static defaultValue = 0.0;

    /**
     * @var string The tooltip to show
     */
    static tooltip = "The amount to change the image when upscaling, from 0 to 1. Keep this low to improve consistency in the upscaled image, or increase it to add many details for a tableau or panorama style. Setting this to zero will skip re-diffusing the upscaled sample.";
}

/**
 * The steps for upscaling
 */
class UpscaleDiffusionStepsInputView extends NumberInputView {
    /**
     * @var int the minimum value
     */
    static min = 0;

    /**
     * @var int The max value
     */
    static max = 200;

    /**
     * @var int The default value
     */
    static defaultValue = 100;
    
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The number of inference steps to make during the denoising loop of the upscaled image. Higher values can result in more details but can also take significantly longer, especially with high denoising strengths.";
}

/**
 * The input view for guidance scale
 */
class UpscaleDiffusionGuidanceScaleInputView extends FloatInputView {
    /**
     * @var float the minimum value
     */
    static min = 1.0;

    /**
     * @var float The max value
     */
    static max = 100.0;

    /**
     * @var float The step value
     */
    static step = 0.01;

    /**
     * @var float The default value
     */
    static defaultValue = 12.0;

    /**
     * @var string The tooltip to show
     */
    static tooltip = "The amount to adhere to the prompts during upscaling. Higher values can result in more details but less consistency.<br />When using multiple guidance scales, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.";
}

/**
 * Allow choosing pipeline to upscale with
 */
class UpscaleDiffusionPipelineInputView extends SelectInputView {
    /**
     * @var object Options
     */
    static defaultOptions = {
        "base": "Always Use Base Pipeline"
    };

    /**
     * @var bool Allow empty
     */
    static allowEmpty = true;

    /**
     * @var string empty text
     */
    static placeholder = "Use Refiner Pipeline when Available";

    /**
     * @var string tooltip
     */
    static tooltip = "When re-diffusing upscaled samples, you can use the base pipeline, or the refiner pipeline, when one is present. Generally the refiner is better tuned for this task, but change this option to always use the base pipeline instead.";
}

export {
    UpscaleAmountInputView,
    UpscaleMethodInputView,
    UpscaleDiffusionControlnetInputView,
    UpscaleDiffusionPromptInputView,
    UpscaleDiffusionNegativePromptInputView,
    UpscaleDiffusionStepsInputView,
    UpscaleDiffusionStrengthInputView,
    UpscaleDiffusionPipelineInputView,
    UpscaleDiffusionGuidanceScaleInputView
};
