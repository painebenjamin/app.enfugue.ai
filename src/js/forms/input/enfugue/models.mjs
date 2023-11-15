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
            } else if (newValue.indexOf("/") !== -1) {
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
        "consistency": "Consistency Decoder",
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
 * Inversion input - will be populated at init.
 */
class InversionInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;
};

/**
 * LoRA input - will be populated at init.
 */
class LoraInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;
};

/**
 * LyCORIS input - will be populated at init.
 */
class LycorisInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;
};

/**
 * Checkpoint input - will be populated at init.
 */
class CheckpointInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;
};

/**
 * Motion module input - will be populated at init.
 */
class MotionModuleInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;
};

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
 * When merging models, there are different modes of operation.
 */
class ModelMergeModeInputView extends SelectInputView {
    /**
     * @var object names and labels
     */
    static defaultOptions = {
        "add-difference": "Add Difference",
        "weighted-sum": "Weighted Sum"
    };
};

/**
 * When download models, this will select the destination
 */
class ModelTypeInputView extends SelectInputView {
    /**
     * @var object names and labels
     */
    static defaultOptions = {
        "checkpoint": "Checkpoint",
        "lora": "LoRA",
        "lycoris": "LyCORIS",
        "inversion": "Textual Inversion",
        "motion": "Motion Module",
    };
};

export {
    CheckpointInputView,
    LoraInputView,
    LycorisInputView,
    InversionInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView,
    VaeInputView,
    DefaultVaeInputView,
    ModelPickerStringInputView,
    ModelPickerListInputView,
    ModelPickerInputView,
    ModelMergeModeInputView,
    MotionModuleInputView,
    ModelTypeInputView,
};
