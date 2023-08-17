/** @module forms/input/enfugue/settings.mjs */
import { FormView } from "../../base.mjs";
import { SelectInputView } from "../enumerable.mjs";

/**
 * Controls how pipelines are switched
 */
class PipelineSwitchModeInputView extends SelectInputView {
    /**
     * @var object The options for switching
     */
    static defaultOptions = {
        "offload": "Offload to CPU",
        "unload": "Unload"
    };
    
    /**
     * @var string Default to offloading to CPU
     */
    static defaultValue = "offload";

    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Keep in Memory";

    /**
     * @var string The tooltip to show the user
     */
    static tooltip = "When making an image, you may switch between a text or infererence pipeline, an inpainting pipeline, and a refining pipeline. In order to balance time spent moving between disk, system memory (RAM), and graphics memory (VRAM), as well as the amount of those resources consumed, the default setting sends pipelines from VRAM to RAM when not needed, then reloads from RAM when needed.<br/><br/>If you set this to <strong>Unload</strong>, pipelines will be freed from VRAM and not sent to RAM when not needed. This will minimize memory consumption, but increase the time spent loading from disk.<br/><br/>If you set this to <strong>Keep in Memory</strong>, pipelines will never be freed from VRAM, and up to three pipelines will be kept available, elimining swapping time. This should only be used with powerful GPUs with <em>at least</em> 12GB of VRAM for SD 1.5 models, or 24GB of VRAM for SDXL models.";
};

/**
 * Controls how pipelines are cached
 */
class PipelineCacheModeInputView extends SelectInputView {
    /**
     * @var object The options for caching
     */
    static defaultOptions = {
        "xl": "Cache XL Pipelines and TensorRT Pipelines",
        "always": "Cache All Pipelines"
    };
    
    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Cache TensorRT Pipelines";

    /**
     * @var string The text to show to the user
     */
    static tooltip = "Models are distributed as <em>.ckpt</em> or <em>.safetensors</em> files for convenience, but opening them and reading them into memory can take some time. For this reason, a cache can be created that speeds up loading time, but takes up approximately as much space as the original checkpoint did, effectively doubling the space a model would take up.<br/><br/>For technical reasons, TensorRT pipelines must always be cached, so this setting cannot be disabled.<br/><br/>The default setting additionally caches when using SDXL models, as this speeds up loading by several factors.<br/><br/>Change this setting to <strong>Cache All Pipelines</strong> in order to cache every pipeline you load. This takes up the most space, but makes switching pipelines the fastest.";
};

/**
 * Controls how data types are changed
 */
class PipelinePrecisionModeInputView extends SelectInputView {
    /**
     * @var object The options for data types
     */
    static defaultOptions = {
        "full": "Always Use Full Precision"
    };
    
    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Use Half-Precision When Available";
    
    /**
     * @var string The text to show to the user
     */
    static tooltip = "When performing calculations on your GPU, we use floating-point numbers of a certain precision. In some cases we must use full-precision in order to calculate correctly, but in some places this is not necessary and calculations can be performed at half-precision instead, without losing quality. In general, you should only change this setting to 'Always Use Full Precision' when you experience errors during diffusion.";
};

export {
    PipelineSwitchModeInputView,
    PipelineCacheModeInputView,
    PipelinePrecisionModeInputView
};
