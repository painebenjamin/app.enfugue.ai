/** @module forms/input/enfugue/noise */
import { SelectInputView } from "../enumerable.mjs";
import { SliderPreciseInputView } from "../numeric.mjs";

/**
 * Define the noise method select
 */
class NoiseMethodInputView extends SelectInputView {
    /**
     * @var object Options and display names
     */
    static defaultOptions = {
        "default": "Default (CPU Random)",
        "crosshatch": "Crosshatch",
        "simplex": "Simplex",
        "perlin": "Perlin",
        "brownian_fractal": "Brownian Fractal",
        "velvet": "Velvet",
        "white": "White",
        "grey": "Grey",
        "pink": "Pink",
        "blue": "Blue",
        "green": "Green",
        "violet": "Violet",
    };

    /**
     * @var string Default value
     */
    static defaultValue = "simplex";

    /**
     * @var string tooltip
     */
    static tooltip = "Each noise method generates randomness through a different method, resulting in varying patterns, shapes and amounts. Some combinations of noise method, blending method and amount will result in illegible images.";
}

/**
 * Define the blend method select
 */
class BlendMethodInputView extends SelectInputView {
    /**
     * @var object Options and display names
     */
    static defaultOptions = {
        "bislerp": "Bilinear Sinc",
        "cosine": "Cosine",
        "cubic": "Cubic",
        "difference": "Difference",
        "inject": "Inject",
        "lerp": "Linear",
        "slerp": "Spherical",
        "exclusion": "Exclusion",
        "subtract": "Subtract",
        "multiply": "Multiply",
        "overlay": "Overlay",
        "screen": "Screen",
        "linear_dodge": "Linear Dodge",
        "glow": "Glow",
        "pin_light": "Pin Light",
        "hard_light": "Hard Light",
        "linear_light": "Linear Light",
        "vivid_light": "Vivid Light"
    };

    /**
     * @var string Default value
     */
    static defaultValue = "inject";

    /**
     * @var string tooltip
     */
    static tooltip = "The blending method controls how noise is added to the initial latents. Methods are similar to blending methods in other photo editing software. Some combinations of noise method, blending method and amount will result in illegible images.";
}

/**
 * Define the noise amount slider
 */
class NoiseOffsetInputView extends SliderPreciseInputView {
    /**
     * @var string Tooltip to display
     */
    static tooltip = "Offset noise can be injected into an image prior to the denoising step in order to add more detail or adjust the base noise that will be used to generate the final image. Some combinations of noise method, blending method and amount will result in illegible images.";
}

export {
    NoiseMethodInputView,
    BlendMethodInputView,
    NoiseOffsetInputView,
};
