/** @module controller/common/model-manager */
import { Controller } from "../base.mjs";
import { ParentView } from "../../view/base.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { TableView, ModelTableView } from "../../view/table.mjs";
import { 
    StringInputView,
    TextInputView,
    FloatInputView,
    NumberInputView,
    FormInputView,
    RepeatableInputView,
    SearchListInputView,
    ButtonInputView
} from "../../view/forms/input.mjs";
import { isEmpty, deepClone } from "../../base/helpers.mjs";

/**
 * Inversion input - will be populated at init.
 */
class InversionInputView extends SearchListInputView {};

/**
 * Lora input - will be populated at init.
 */
class LoraInputView extends SearchListInputView {};

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
 * The overall multi-input that allows any number of lora
 */
class MultiLoraInputView extends RepeatableInputView {
    /**
     * @var class The repeatable input element.
     */
    static memberClass = LoraFormInputView;
};

/**
 * The overall multi-input that allows any number of inversions
 */
class MultiInversionInputView extends RepeatableInputView {
    /**
     * @var class The repeatable input element.
     */
    static memberClass = InversionInputView;
};

/**
 * The model form pulls it all together for making/editing models
 */
class ModelForm extends FormView {
    /**
     * @var bool Enable cancel
     */
    static canCancel = true;

    /**
     * @var object All fieldsets; labels are preserved for parent forms.
     */
    static fieldSets = {
        "Name": {
            "name": {
                "class": StringInputView,
                "label": "Name",
                "config": {
                    "required": true
                }
            }
        },
        "Model": {
            "checkpoint": {
                "class": CheckpointInputView,
                "label": "Checkpoint",
                "config": {
                    "required": true
                }
            },
            "lora": {
                "class": MultiLoraInputView,
                "label": "LoRA"
            },
            "inversion": {
                "class": MultiInversionInputView,
                "label": "Textual Inversions"
            }
        },
        "Engine": {
            "size": {
                "class": NumberInputView,
                "label": "Size",
                "config": {
                    "required": true,
                    "value": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 8
                },
            }
        },
        "Prompts": {
            "prompt": {
                "class": TextInputView,
                "label": "Prompt"
            },
            "negative_prompt": {
                "class": TextInputView,
                "label": "Negative Prompt"
            }
        }
    };
};

/**
 * New model input is a simple extension of the buttonview
 */
class NewModelInputView extends ButtonInputView {
    /**
     * @var string The value to display in the button
     */
    static defaultValue = "New Model";

    /**
     * @var string The class name for CSS
     */
    static className = "new-model-input-view";
};

/**
 * This shared controller makes it easy to view/edit 
 * models from anywhere else in the application.
 */
class ModelManagerController extends Controller {
    /**
     * @var int The starting width of the model form window
     */
    static modelWindowWidth = 400;
    
    /**
     * @var int The starting height of the model form window
     */
    static modelWindowHeight = 1000;

    /**
     * @var int The starting width of the manager form window
     */
    static managerWindowWidth = 800;
    
    /**
     * @var int The starting height the manager form window
     */
    static managerWindowHeight = 600;

    /**
     * Creates the manager table.
     */
    async createManager() {
        this.tableView = new ModelTableView(this.config, this.model.DiffusionModel);
        this.buttonView = new NewModelInputView(this.config);

        this.tableView.setColumns({
            "name": "Name",
            "model": "Model",
            "size": "Size",
            "prompt": "Prompt",
            "negative_prompt": "Negative Prompt"
        });
        this.tableView.setFormatter("size", (datum) => `${datum}px`);
        
        // Add the 'Edit' button
        this.tableView.addButton("Edit", "fa-solid fa-edit", async (row) => {
            let modelValues = row.getAttributes();
            modelValues.checkpoint = modelValues.model;
            modelValues.lora = isEmpty(row.lora) ? [] : row.lora.map((lora) => lora.getAttributes());
            modelValues.inversion = isEmpty(row.inversion) ? [] : row.inversion.map((inversion) => inversion.model);

            let modelForm = new ModelForm(this.config, deepClone(modelValues)),
                modelWindow;
    
            modelForm.onSubmit(async (updatedValues) => {
                try {
                    await this.model.put(`/models/${row.name}`, null, null, updatedValues);
                    if (!isEmpty(modelWindow)) {
                        modelWindow.remove();
                    }
                    this.tableView.requery();
                } catch(e) {
                    let errorMessage = isEmpty(e)
                        ? "Couldn't communicate with server."
                        : isEmpty(e.detail)
                            ? `${e}`
                            : e.detail;

                    this.notify("error", "Couldn't update model", errorMessage);
                    modelForm.enable();
                }
            });
            modelForm.onCancel(() => modelWindow.remove());
            modelWindow = await this.spawnWindow(
                `Edit ${row.name}`,
                modelForm,
                this.constructor.modelWindowWidth,
                this.constructor.modelWindowHeight
            );
        });
        
        // Add the 'Delete' button
        this.tableView.addButton("Delete", "fa-solid fa-trash", async (row) => {
            try{
                await this.model.delete(`/models/${row.name}`);
                this.tableView.requery();
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail)
                        ? `${e}`
                        : e.detail;

                this.notify("error", "Couldn't delete model", errorMessage);
                modelForm.enable();
            }
        });

        this.buttonView.onChange(() => {
            this.showNewModel();
        });

        let managerView = new ParentView(this.config);
        managerView.addChild(this.tableView);
        managerView.addChild(this.buttonView);
        return managerView;
    }

    /**
     * Creates the 'New Model' form.
     */
    async createModelForm() {
        let modelForm = new ModelForm(this.config);
        modelForm.onSubmit(async (values) => {
            try {
                let response = await this.model.post("/models", null, null, values);
                if (!isEmpty(this.newModelWindow)) {
                    this.newModelWindow.remove();
                    this.newModelWindow = null;
                }
                if (!isEmpty(this.managerWindow) && !isEmpty(this.tableView)) {
                    this.tableView.requery();
                }
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail) 
                        ? `${e}` 
                        : e.detail;
                this.notify("Error", "Couldn't create model", errorMessage);
            }
        });
        modelForm.onCancel(() => {
            this.newModelWindow.remove();
            this.newModelWindow = null;
        });
        return modelForm;
    }
    
    /**
     * Shows the New Model form.
     */
    async showNewModel() {
        if (!isEmpty(this.newModelWindow)) {
            this.newModelWindow.focus();
        } else {
            this.newModelWindow = await this.spawnWindow(
                "New Model",
                this.createModelForm(),
                this.constructor.modelWindowWidth,
                this.constructor.modelWindowHeight
            );
            this.newModelWindow.onClose(() => { delete this.newModelWindow; });
        }
    }

    /**
     * Shows the model manager table.
     */
    async showManager() {
        if (!isEmpty(this.managerWindow)) {
            this.managerWindow.focus();
        } else {
            this.managerWindow = await this.spawnWindow(
                "Model Manager",
                this.createManager(),
                this.constructor.managerWindowWidth,
                this.constructor.managerWindowHeight
            );
            this.managerWindow.onClose(() => { delete this.managerWindow; });
        }
    }

    /**
     * On initialization, set option getters.
     */
    async initialize() {
        LoraInputView.defaultOptions = async () => this.model.get("/lora");
        CheckpointInputView.defaultOptions = async () => this.model.get("/checkpoints");
        InversionInputView.defaultOptions = async () => this.model.get("/inversions");
    }
}

export { ModelManagerController };
