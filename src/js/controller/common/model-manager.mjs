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
     * @var class The repeatable input element.
     */
    static memberClass = LoraFormInputView;
};

/**
 * The overall multi-input that allows any number of lycoris
 */
class MultiLycorisInputView extends RepeatableInputView {
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
                    "required": true,
                    "tooltip": "Give your model a name that describes what you want it to do - for example, if you're using a photorealistic model and use phrases related to central framing, bokeh focus and and saturated colors, you could call this configuration &ldquo;Product Photography.%rdquo;"
                }
            }
        },
        "Model": {
            "checkpoint": {
                "class": CheckpointInputView,
                "label": "Checkpoint",
                "config": {
                    "required": true,
                    "tooltip": "A &ldquo;checkpoint&rdquo; represents the state of the Stable Diffusion model at a given point in it's training. Generally, checkpoints are started from a particular version of the foundation Stable Diffusion model (1.5, 2.1, XL 1.0, etc.) and fine-tuned on a particular style or subject of imagery, though you can also use the foundation checkpoints on their own."
                }
            },
        },
        "Adaptations and Modifications": {
            "lora": {
                "class": MultiLoraInputView,
                "label": "LoRA",
                "config": {
                    "tooltip": "LoRA stands for <strong>Low Rank Adapation</strong>, it is a kind of fine-tuning that can perform very specific modifications to Stable Diffusion such as training an individual's appearance, new products that are not in Stable Diffusion's training set, etc."
                }
            },
            "lycoris": {
                "class": MultiLycorisInputView,
                "label": "LyCORIS",
                "config": {
                    "tooltip": "LyCORIS stands for <strong>LoRA beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion</strong>, a novel means of performing low-rank adaptation introduced in early 2023."
                }
            },
            "inversion": {
                "class": MultiInversionInputView,
                "label": "Textual Inversions",
                "config": {
                    "tooltip": "Textual Inversion is another kind of fine-tuning that teaches novel concepts to Stable Diffusion in a small number of images, which can be used to positively or negatively affect the impact of various prompts."
                }
            }
        },
        "Additional Models": {
            "refiner": {
                "class": CheckpointInputView,
                "label": "Refining Checkpoint",
                "config": {
                    "tooltip": "Refiner checkpoints were introduced with SDXL 0.9 - these are checkpoints specifically trained to improve detail, shapes, and generally improve the quality of images generated from the base model. These are optional, and do not need to be specifically-trained refinement checkpoints - you can try mixing and matching checkpoints for different styles, though you may wish to ensure the related checkpoints were trained on the same size images."
                }
            },
            "inpainter": {
                "class": CheckpointInputView,
                "label": "Inpainting Checkpoint",
                "config": {
                    "tooltip": "An inpainting checkpoint if much like a regular Stable Diffusion checkpoint, but it additionally includes the ability to input which parts of the image can be changed and which cannot. This is used when you specifically request an image be inpainted, but is also used in many other situations in Enfugue; such as when you place an image on the canvas that doesn't cover the entire space, or use an image that has transparency in it (either before or after removing it's background.) When you don't select an inpainting checkpoint and request an inpainting operation, one will be created dynamically from the main checkpoint at runtime."
                }
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
                    "step": 8,
                    "tooltip": "When using chunked diffusion, this is the size of the window (in pixels) that will be encoded, decoded or inferred at once. Set the chunking size to 0 in the sidebar to disable chunked diffusion and always try to process the entire image at once."
                }
            }
        },
        "Prompts": {
            "prompt": {
                "class": TextInputView,
                "label": "Prompt",
                "tooltip": "This prompt will be appended to every prompt you make when using this model. Use this field to add trigger words, style or quality phrases that you always want to be included."
            },
            "negative_prompt": {
                "class": TextInputView,
                "label": "Negative Prompt",
                "tooltip": "This prompt will be appended to every negative prompt you make when using this model. Use this field to add trigger words, style or quality phrases that you always want to be excluded."
            }
        }
    };

    static collapseFieldSets = [
        "Adaptations and Modifications", "Additional Models"
    ];
};

/**
 * New model input is a simple extension of the buttonview
 */
class NewModelInputView extends ButtonInputView {
    /**
     * @var string The value to display in the button
     */
    static defaultValue = "New Model Configuration";

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
            console.log(row);
            let modelValues = row.getAttributes();
            modelValues.checkpoint = modelValues.model;
            modelValues.lora = isEmpty(row.lora) ? [] : row.lora.map((lora) => lora.getAttributes());
            modelValues.lycoris = isEmpty(row.lycoris) ? [] : row.lycoris.map((lycoris) => lycoris.getAttributes());
            modelValues.inversion = isEmpty(row.inversion) ? [] : row.inversion.map((inversion) => inversion.model);
            modelValues.refiner = isEmpty(row.refiner) ? null : row.refiner[0].model;
            modelValues.inpainter = isEmpty(row.inpainter) ? null : row.inpainter[0].model;

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
                "New Configuration",
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
                "Model Configuration Manager",
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
        LycorisInputView.defaultOptions = async () => this.model.get("/lycoris");
        CheckpointInputView.defaultOptions = async () => this.model.get("/checkpoints");
        InversionInputView.defaultOptions = async () => this.model.get("/inversions");
    }
}

export {
    ModelManagerController,
    CheckpointInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView
};
