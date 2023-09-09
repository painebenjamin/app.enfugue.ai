/** @module forms/input/enfugue/models */
import { isEmpty, deepClone, createElementsFromString } from "../../../base/helpers.mjs";
import { FormView } from "../../base.mjs";
import { InputView } from "../base.mjs";
import { StringInputView, TextInputView } from "../string.mjs";
import { NumberInputView, FloatInputView } from "../numeric.mjs";
import { FormInputView, RepeatableInputView } from "../parent.mjs";
import {
    SelectInputView,
    SearchListInputView,
    SearchListInputListView
} from "../enumerable.mjs";

let defaultEngineSize = 512;

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
     * @var int The default value
     */
    static defaultValue = defaultEngineSize;
    
    /**
     * @var string The tooltip to display to the user
     */
    static tooltip = "When using chunked diffusion, this is the size of the window (in pixels) that will be encoded, decoded or inferred at once. Set the chunking size to 0 in the sidebar to disable chunked diffusion and always try to process the entire image at once.";
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
 * This class shows the default VAE's and allows an other option
 */
class VaeInputView extends InputView {
    /**
     * @var Custom tag name
     */
    static tagName = "enfugue-vae-input-view";

    /**
     * @var class VAE input class
     */
    static selectClass = DefaultVaeInputView;

    /**
     * @var class text input class
     */
    static textClass = StringInputView;

    /**
     * @var object Text input config
     */
    static textInputConfig = {
        "placeholder": "e.g. stabilityai/sdxl-vae",
        "tooltip": "Enter the name of a HuggingFace repository housing the VAE configuration. Visit https://huggingface.co for more information."
    };

    /**
     * On construct, instantiate sub inputs
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.defaultInput = new this.constructor.selectClass(config, "default");
        this.otherInput = new this.constructor.textClass(config, "other", this.constructor.textInputConfig);
        this.defaultInput.onChange(() => {
            let value = this.defaultInput.getValue();
            if (value === "other") {
                this.value = "";
                this.otherInput.show();
            } else {
                this.value = value;
                this.otherInput.hide();
            }
            this.changed();
        });
        this.otherInput.onChange(() => {
            if (this.defaultInput.getValue() === "other") {
                this.value === this.otherInput.getValue();
                this.changed();
            }
        });
        this.otherInput.hide();
    }

    /**
     * Get value from inputs
     */
    getValue() {
        let defaultValue = this.defaultInput.getValue();
        if (defaultValue === "other") {
            return this.otherInput.getValue();
        }
        return defaultValue;
    }

    /**
     * Sets the value in sub inputs
     */
    setValue(newValue, triggerChange) {
        super.setValue(newValue, false);
        if (isEmpty(newValue)) {
            this.defaultInput.setValue(null, false);
            this.otherInput.setValue("", false);
            this.otherInput.hide();
        } else if (Object.getOwnPropertyNames(DefaultVaeInputView.defaultOptions).indexOf(newValue) === -1) {
            this.defaultInput.setValue("other", false);
            this.otherInput.setValue(newValue, false);
            this.otherInput.show();
        } else {
            this.defaultInput.setValue(newValue, false);
            this.otherInput.setValue("", false);
            this.otherInput.hide();
        }
        if (triggerChange) {
            this.changed();
        }
    }

    /**
     * On build, get both inputs.
     */
    async build() {
        let node = await super.build();
        return node.content(
            await this.defaultInput.getNode(),
            await this.otherInput.getNode()
        );
    }
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
    static tooltip = "During multi-diffusion, only a square of the size of the engine is rendereda at any given time. This can cause hard edges between the frames, especially when using a large chunking size. Using a mask allows for blending along the edges - this can remove seams, but also reduce precision.";

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
        "dpmsm": "DPM-Solver++ Multi-Step",
        "dpmss": "DPM-Solver++ Single-Step",
        "heun": "Heun Discrete Scheduler",
        "dpmd": "DPM Discrete Scheduler",
        "adpmd": "DPM Ancestral Discrete Scheduler",
        "dpmsde": "DPM Solver SDE Scheduler",
        "unipc": "UniPC: Predictor (UniP) and Corrector (UniC)",
        "lmsd": "LMS: Linear Multi-Step Discrete Scheduler",
        "pndm": "PNDM: Pseudo Numerical Methods for Diffusion Models",
        "eds": "Euler Discrete Scheduler",
        "eads": "Euler Ancestral Discrete Scheduler",
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
 * Inversion input - will be populated at init.
 */
class InversionInputView extends SearchListInputView {};

/**
 * LoRA input - will be populated at init.
 */
class LoraInputView extends SearchListInputView {};

/**
 * LyCORIS input - will be populated at init.
 */
class LycorisInputView extends SearchListInputView {};

/**
 * Checkpoint input - will be populated at init.
 */
class CheckpointInputView extends SearchListInputView {};

/**
 * Lora input additionally has weight; create the FormView here,
 * then define a RepeatableInputView of a FormInputView
 */
class LoraFormView extends FormView {
    /**
     * @var bool disable submit button for form, automatically submit on every change
     */
    static autoSubmit = true;

    /**
     * @var object All fieldsets; the label will be removed.
     */
    static fieldSets = {
        "LoRA": {
            "model": {
                "label": "Model",
                "class": LoraInputView,
                "config": {
                    "required": true
                }
            },
            "weight": {
                "label": "Weight",
                "class": FloatInputView,
                "config": {
                    "min": 0,
                    "value": 1.0,
                    "step": 0.01,
                    "required": true
                }
            }
        }
    };
};

/**
 * The input element containing the parent form
 */
class LoraFormInputView extends FormInputView {
    /**
     * @var class The sub-form to use in the input.
     */
    static formClass = LoraFormView;
};

/**
 * Lycoris input additionally has weight; create the FormView here,
 * then define a RepeatableInputView of a FormInputView
 */
class LycorisFormView extends FormView {
    /**
     * @var bool disable submit button for form, automatically submit on every change
     */
    static autoSubmit = true;

    /**
     * @var object All fieldsets; the label will be removed.
     */
    static fieldSets = {
        "LyCORIS": {
            "model": {
                "label": "Model",
                "class": LycorisInputView,
                "config": {
                    "required": true
                }
            },
            "weight": {
                "label": "Weight",
                "class": FloatInputView,
                "config": {
                    "min": 0,
                    "value": 1.0,
                    "step": 0.01,
                    "required": true
                }
            }
        }
    };
};

/**
 * The input element containing the parent form
 */
class LycorisFormInputView extends FormInputView {
    /**
     * @var class The sub-form to use in the input.
     */
    static formClass = LycorisFormView;
};

/**
 * The overall multi-input that allows any number of lora
 */
class MultiLoraInputView extends RepeatableInputView {
    /**
     * @var string Text to display when no items are added
     */
    static noItemsLabel = "No LoRA Configured";

    /**
     * @var string Text to show in the add items buttons
     */
    static addItemLabel = "Add LoRA";

    /**
     * @var class The repeatable input element.
     */
    static memberClass = LoraFormInputView;
};

/**
 * The overall multi-input that allows any number of lycoris
 */
class MultiLycorisInputView extends RepeatableInputView {
    /**
     * @var string Text to display when no items are added
     */
    static noItemsLabel = "No LyCORIS Configured";

    /**
     * @var string Text to show in the add items buttons
     */
    static addItemLabel = "Add LyCORIS";

    /**
     * @var class The repeatable input element.
     */
    static memberClass = LycorisFormInputView;
};

/**
 * The overall multi-input that allows any number of inversions
 */
class MultiInversionInputView extends RepeatableInputView {
    /**
     * @var string Text to display when no items are added
     */
    static noItemsLabel = "No Textual Inversion Configured";

    /**
     * @var string Text to show in the add items buttons
     */
    static addItemLabel = "Add Textual Inversion";

    /**
     * @var class The repeatable input element.
     */
    static memberClass = InversionInputView;
};

/**
 * Extend the SearchListInputListView to add additional classes
 */
class ModelPickerListInputView extends SearchListInputListView {
    /**
     * @var array<string> CSS classes
     */
    static classList = SearchListInputListView.classList.concat(["model-picker-list-input-view"]);
};

/**
 * Extend the StringInputView so we can strip HTML from the value
 */
class ModelPickerStringInputView extends StringInputView {
    /**
     * Strip HTML from the value and only display the name portion.
     */
    setValue(newValue, triggerChange) {
        if(!isEmpty(newValue)) {
            if (newValue.startsWith("<")) {
                newValue = createElementsFromString(newValue)[0].innerText;
            } else {
                newValue = newValue.split("/")[1];
            }
        }
        return super.setValue(newValue, triggerChange);
    }
};

/**
 * We extend the SearchListInputView to change some default config.
 */
class ModelPickerInputView extends SearchListInputView {
    /**
     * @var string The content of the node when nothing is selected.
     */
    static placeholder = "Start typing to search models…";

    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;

    /**
     * @var class The class of the list input, override so we can add css classes
     */
    static listInputClass = ModelPickerListInputView;
};

export {
    CheckpointInputView,
    LoraInputView,
    LycorisInputView,
    InversionInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView,
    EngineSizeInputView,
    RefinerEngineSizeInputView,
    InpainterEngineSizeInputView,
    VaeInputView,
    DefaultVaeInputView,
    SchedulerInputView,
    ModelPickerStringInputView,
    ModelPickerListInputView,
    ModelPickerInputView,
    MaskTypeInputView
};
