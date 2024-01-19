/** @module forms/input/enfugue/models */
import { isEmpty, deepClone, createElementsFromString } from "../../../base/helpers.mjs";
import { ElementBuilder } from "../../../base/builder.mjs";
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

const E = new ElementBuilder();

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

    /**
     * Shows metadata in a separate window
     */
    async showMetadata() {
        this.showingMetadata = true;
        try {
            await this.constructor.showModelMetadata(this.value);
        } finally {
            this.showingMetadata = false;
        }
    }

    /**
     * Sets the value and shows/hides the icon
     */
    setValue(newValue, triggerChange) {
        super.setValue(newValue, triggerChange);
        if (!triggerChange) {
            if (this.node !== undefined) {
                let icon = this.node.find("i.show-metadata");
                if (isEmpty(newValue) || !newValue.startsWith("checkpoint")) {
                    icon.hide();
                } else {
                    icon.show();
                }
            }
        }
    }

    /**
     * On build, append introspection button
     */
    async build() {
        let node = await super.build(),
            icon = E.i().class("fa-solid fa-magnifying-glass show-metadata")
                    .data("tooltip", "View Model Metadata")
                    .on("click", (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        if (this.showingMetadata !== true) {
                            this.showMetadata();
                        }
                    });

        node.append(icon);

        this.onChange(() => {
            if (isEmpty(this.value) || !this.value.startsWith("checkpoint")) {
                icon.hide();
            } else {
                icon.show();
            }
        });
        if (isEmpty(this.value)) {
            icon.hide();
        }
        return node;
    }
};

/**
 * A superclass for introspectable models
 */
class ModelInputView extends SearchListInputView {
    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;

    /**
     * On set value, hide/show
     */
     async setValue(newValue) {
        await super.setValue(newValue);
        if (this.node !== undefined) {
            let icon = this.node.find(".show-metadata");
            if (isEmpty(this.value)) {
                icon.hide();
            } else {
                icon.show();
            }
        }
    }

    /**
     * Shows metadata in a separate window
     */
    async showMetadata() {
        this.showingMetadata = true;
        try {
            await this.constructor.showModelMetadata(this.value);
        } finally {
            this.showingMetadata = false;
        }
    }

    /**
     * On build, append introspection button
     */
    async build() {
        let node = await super.build(),
            icon = E.i().class("fa-solid fa-magnifying-glass show-metadata")
                    .data("tooltip", "View Model Metadata")
                    .on("click", (e) => {
                        e.stopPropagation();
                        e.preventDefault();
                        if (this.showingMetadata !== true) {
                            this.showMetadata();
                        }
                    });

        node.append(icon);

        this.onChange(() => {
            if (isEmpty(this.value)) {
                icon.hide();
            } else {
                icon.show();
            }
        });
        if (isEmpty(this.value)) {
            icon.hide();
        }
        return node;
    }
}

/**
 * LoRA input - will be populated at init.
 */
class LoraInputView extends ModelInputView { };

/**
 * VAE input - will be populated at init.
 */
class VAEInputView extends ModelInputView { };

/**
 * LyCORIS input - will be populated at init.
 */
class LycorisInputView extends ModelInputView { };

/**
 * Inversion input - will be populated at init.
 */
class InversionInputView extends ModelInputView { };

/**
 * Checkpoint input - will be populated at init.
 */
class CheckpointInputView extends ModelInputView { };

/**
 * Motion module input - will be populated at init.
 */
class MotionModuleInputView extends ModelInputView { };

/**
 * ControlNet model input - will be populated at init.
 */
class ControlNetModelInputView extends ModelInputView { };

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
    VAEInputView,
    ModelPickerStringInputView,
    ModelPickerListInputView,
    ModelPickerInputView,
    ModelMergeModeInputView,
    MotionModuleInputView,
    ModelTypeInputView,
    ControlNetModelInputView,
};
