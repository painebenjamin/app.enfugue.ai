/** @module forms/input/enfugue/upscale */
import { SelectInputView } from "../enumerable.mjs";
import { RepeatableInputView } from "../parent.mjs";
import { NumberInputView, FloatInputView } from "../numeric.mjs";
import { PromptInputView } from "./prompts.mjs";

/**
 * The select box for outscale
 * 16x at 512 is 8k, if we allow much bigger it'll really bog down a browser,
 * even on the best of systems.
 */
class OutputScaleInputView extends SelectInputView {
    /**
     * @var object The option values and label
     */
    static defaultOptions = {
        "1": "1× (no upscale)",
        "2": "2×",
        "4": "4×",
        "8": "8×",
        "16": "16×"
    };

    /**
     * @var string The tooltip to display to the user
     */
    static tooltip = "The output scale will multiply the height and width of the generated image by this amount after the image has been generated. For example, an image generated at 512×512 with an output scale of 2× will result in a final image at 1024×1024.<br /><strong>Caution!</strong> Large values, especially coupled with larger input sizes, can result in an image that will be too large for your browser to display, and it will crash. The resulting images are still saved.";

    /**
     * @var string The value, keep as string for compat
     */
    static defaultValue = "1";
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
        "mlsd": "MLSD (Mobile Line Segment Detection)"
    };
    
    /**
     * @var string The default value
     */
    static defaultValue = "tile";
}

/**
 * The repeatable input view for method
 * We let them choose different methods for different iterations
 */
class UpscaleMethodsInputView extends RepeatableInputView {
    /**
     * @var int Always have one value when this is visible
     */
    static minimumItems = 1;

    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;
    
    /**
     * @var class The repeatable item class
     */
    static memberClass = UpscaleMethodInputView;

    /**
     * @var string The tooltip to display
     */
    static tooltip = "The upscaling method has a significant effect on the output image. The best general-purpose upscaling method is selected by default.<br />When selecting multiple methods, the first is used for the first upscale, the second for the second (when using iterative upscaling), etc.<br /><strong>ESRGAN</strong>: Short for Enhanced Super-Resolution Generative Adversarial Network, this is an AI upscaling method that tries to maintain sharp edges where they should be sharp, and soft where they should be soft, filling in details along the way.<br /><strong>ESRGANime</strong>: Similar to the above, but with sharper lines for cartoon or anime style.<br /><strong>GFPGAN</strong>: Short for Generative Facial Prior Generative Adversarial Network, this is an AI Upscaling method with face restoration. This results in photorealistic faces more often than not, but can erase desired features; it is best paired with upscale diffusion.<br /><strong>Lanczos</strong>: An algorithm with blurry but consistent results.<br /><strong>Bicubic</strong>: An algorithm that can result in slightly sharper edges than Lanczos, but can have jagged edges on curves and diagonal lines.<br /><strong>Bilinear</strong>: A very fast algorithm with overall the blurriest results.<br /><strong>Nearest</strong>: Maintain sharp pixel boundaries, resulting in a pixelated or retro look.";

}

/**
 * The repeatable input view for controlnet
 * None are required, default is none for speed
 */
class UpscaleDiffusionIterativeControlnetInputView extends RepeatableInputView {
    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;

    /**
     * @var int No minimum
     */
    static minimumItems = 0;
    
    /**
     * @var class The repeatable item class
     */
    static memberClass = UpscaleDiffusionControlnetInputView;
    
    /**
     * @var string The tooltip to display
     */
    static tooltip = "The controlnet to use during upscaling. None are required, and using one will result in significant slowdowns during upscaling, but can result in a more consistent upscaled image. When using multiple methods, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.<br /><strong>Tile</strong>: This network is trained on large images and slices of their images.<br /><strong>Canny Edge</strong>: This network is trained on images and the edges of that image after having run through Canny Edge detection. The output image will be processed with this algorithm.<br /><strong>HED</strong>: Short for Holistically-Nested Edge Detection, this edge-detection algorithm is best used when the input image is too blurry or too noisy for Canny Edge detection.<br /><strong>MLSD</strong>: Short for Mobile Line Segment Detection, this edge-detection algorithm searches only for straight lines, and is best used for geometric or architectural images.";
}

/**
 * The repetable input view for prompt
 * None are required, we'll use a generic detail prompt
 */
class UpscaleDiffusionPromptInputView extends RepeatableInputView {
    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;
    
    /**
     * @var class The repeatable item class
     */
    static memberClass = PromptInputView;

    /**
     * @var string The tooltip to show
     */
    static tooltip = "The prompt to use when upscaling, it is generally best to use generic detail-oriented prompts, unless there are specific things or people you want to ensure have details.<br />When using multiple prompts, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.";
}

/**
 * The repetable input view for negative prompt
 * None are required, we'll use a generic detail prompt
 */
class UpscaleDiffusionNegativePromptInputView extends UpscaleDiffusionPromptInputView {
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The negative prompt to use when upscaling, it is generally best to use generic negative prompts, unless there are specific things you don't want.<br />When using multiple prompts, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.";
}

/**
 * The repeatable input view for strength
 * At least one is required
 */
class UpscaleDiffusionStrengthInputView extends RepeatableInputView {
    /**
     * @var class The repeatable item class
     */
    static memberClass = FloatInputView;

    /**
     * @var object The config to pass to the member class
     */
    static memberConfig = {
        "min": 0.0,
        "value": 0.2,
        "max": 1.0,
        "step": 0.01
    };
    
    /**
     * @var int One strength always required
     */
    static minimumItems = 1;
    
    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;
    
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The amount to change the image when upscaling, from 0 to 1. Keep this low to improve consistency in the upscaled image, or increase it to add many details for a tableau or panorama style.<br />When using multiple strengths, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.";
}

/**
 * The repeatable input view for steps
 * At least one is required
 */
class UpscaleDiffusionStepsInputView extends RepeatableInputView {
    /**
     * @var class The repeatable item class
     */
    static memberClass = NumberInputView;

    /**
     * @var object The config to pass to the member class
     */
    static memberConfig = {
        "min": 0,
        "max": 1000,
        "value": 100
    };
    
    /**
     * @var int at least one is required
     */
    static minimumItems = 1;
    
    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;
    
    /**
     * @var string The tooltip to show
     */
    static tooltip = "The number of inference steps to make during the denoising loop of the upscaled image. Higher values can result in more details but can also take significantly longer, especially with high denoising strengths.<br />When using multiple step amounts, the first is used for the first upscale, the second is used for the second (when using iterative upscaling), etc.";
}

/**
 * The repeatable input view for guidance scale
 * At least one is required
 */
class UpscaleDiffusionGuidanceScaleInputView extends RepeatableInputView {
    /**
     * @var class The repetable item class
     */
    static memberClass = FloatInputView;

    /**
     * @var object The config to pass to the class
     */
    static memberConfig = {
        "min": 0.0,
        "max": 100.0,
        "value": 12.0,
        "step": 0.1
    };

    /**
     * @var int at least one is required
     */
    static minimumItems = 1;
    
    /**
     * @var int max upscale is 2^5
     */
    static maximumItems = 5;
    
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
    OutputScaleInputView,
    UpscaleMethodsInputView,
    UpscaleDiffusionIterativeControlnetInputView,
    UpscaleDiffusionPromptInputView,
    UpscaleDiffusionNegativePromptInputView,
    UpscaleDiffusionStepsInputView,
    UpscaleDiffusionStrengthInputView,
    UpscaleDiffusionPipelineInputView,
    UpscaleDiffusionGuidanceScaleInputView
};
