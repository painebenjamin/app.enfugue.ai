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
    ButtonInputView,
    SelectInputView
} from "../../view/forms/input.mjs";
import { isEmpty, deepClone } from "../../base/helpers.mjs";

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
 * VAE Input View
 */
class VAEInputView extends SelectInputView {
    /**
     * @var object Option values and labels
     */
    static defaultOptions = {
        "ema": "EMA 560000",
        "mse": "MSE 840000",
        "xl": "SDXL"
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
 * Limit options for multidiffusion scheduler
 */
class MultiDiffusionSchedulerInputView extends SelectInputView {
    /**
     * @var object Option values and labels
     */
    static defaultOptions = {
        "ddim": "DDIM: Denoising Diffusion Implicit Models (Recommended)",
        "eds": "Euler Discrete Scheduler (Recommended)",
        "ddpm": "DDPM: Denoising Diffusion Probabilistic Models (Blurrier)",
        "eads": "Euler Ancestral Discrete Scheduler (Blurrier)",
        "deis": "DEIS: Diffusion Exponential Integrator Sampler (Distorted)",
        "dpmsm": "DPM-Solver++ Multi-Step (Distorted)",
        "dpmss": "DPM-Solver++ Single-Step (Distorted)",
    };

    /**
     * @var string The tooltip
     */
    static tooltip = "During chunked diffusion (also called multi-diffusion or sliced diffusion,) each denoising step is performed multiple times over different windows of the image. This necessitates that the scheduler be capable of stepping backward as well as forward, and not all schedulers were designed with this in mind. The schedulers in this list are supported during multi-diffusion, but only two are recommended: DDIM, which is the default scheduler for SD 1.5, and Euler Discrete, which is the default scheduler for SDXL.";
    
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
            "vae": {
                "class": VAEInputView,
                "label": "VAE"
            },
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
                "class": EngineSizeInputView,
                "label": "Size",
                "config": {
                    "required": true,
                }
            },
            "refiner_size": {
                "class": RefinerEngineSizeInputView,
                "label": "Refiner Size"
            },
            "inpainter_size": {
                "class": InpainterEngineSizeInputView,
                "label": "Inpainter Size"
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
        },
        "Additional Defaults": {
            "scheduler": {
                "class": SchedulerInputView,
                "label": "Scheduler"
            },
            "multi_scheduler": {
                "class": MultiDiffusionSchedulerInputView,
                "label": "Multi-Diffusion Scheduler"
            },
            "width": {
                "label": "Width",
                "class": NumberInputView,
                "config": {
                    "tooltip": "The width of the canvas in pixels.",
                    "min": 128,
                    "max": 4096,
                    "step": 8,
                    "value": null
                }
            },
            "height": {
                "label": "Height",
                "class": NumberInputView,
                "config": {
                    "tooltip": "The height of the canvas in pixels.",
                    "min": 128,
                    "max": 4096,
                    "step": 8,
                    "value": null
                }
            },
            "chunking_size": {
                "label": "Chunk Size",
                "class": NumberInputView,
                "config": {
                    "tooltip": "<p>The number of pixels to move the frame when doing chunked diffusion.</p><p>When this number is greater than 0, the engine will only ever process a square in the size of the configured model size at once. After each square, the frame will be moved by this many pixels along either the horizontal or vertical axis, and then the image is re-diffused. When this number is 0, chunking is disabled, and the entire canvas will be diffused at once.</p><p>Disabling this (setting it to 0) can have varying visual results, but a guaranteed result is drastically increased VRAM usage for large images. A low number can produce more detailed results, but can be noisy, and takes longer to process. A high number is faster to process, but can have poor results especially along frame boundaries. The recommended value is set by default.</p>",
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "value": null
                }
            },
            "chunking_blur": {
                "label": "Chunk Blur",
                "class": NumberInputView,
                "config": {
                    "tooltip": "The number of pixels to feather along the edge of the frame when blending chunked diffusions together. Low numbers can produce less blurry but more noisy results, and can potentially result in visible breaks in the frame. High numbers can help blend frames, but produce blurrier results. The recommended value is set by default.",
                    "min": 0,
                    "max": 2048,
                    "step": 8,
                    "value": null
                }
            },
            "inference_steps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process but can produce better results.",
                    "min": 0,
                    "step": 1,
                    "value": null
                }
            },
            "guidance_scale": {
                "label": "Guidance Scale",
                "class": NumberInputView,
                "config": {
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness.",
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                    "value": null
                }
            },
            "refiner_denoising_strength": {
                "label": "Refiner Denoising Strength",
                "class": NumberInputView,
                "config": {
                    "tooltip": "When using a refiner, this will control how much of the original image is kept, and how much of it is replaced with refined content. A value of 1.0 represents total destruction of the first image.",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": null
                }
            },
            "refiner_guidance_scale": {
                "label": "Refiner Guidance Scale",
                "class": NumberInputView,
                "config": {
                    "tooltip": "When using a refiner, this will control how closely to follow the guidance of the model. Low values can result in soft details, whereas high values can result in high-contrast ones.",
                    "min": 0,
                    "max": 100,
                    "step": 0.01,
                    "value": null
                }
            },
            "refiner_aesthetic_score": {
                "label": "Refiner Aesthetic Score",
                "class": NumberInputView,
                "config": {
                    "tooltip": "Aesthetic scores are assigned to images in SDXL refinement; this controls the positive score.",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "value": null
                }
            },
            "refiner_negative_aesthetic_score": {
                "label": "Negative Aesthetic Score",
                "class": NumberInputView,
                "config": {
                    "tooltip": "Aesthetic scores are assigned to images in SDXL refinement; this controls the negative score.",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "value": null
                }
            }
        }
    };

    static collapseFieldSets = [
        "Adaptations and Modifications",
        "Additional Models",
        "Additional Defaults"
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
            let modelValues = row.getAttributes();
            modelValues.checkpoint = modelValues.model;
            modelValues.lora = isEmpty(row.lora) ? [] : row.lora.map((lora) => lora.getAttributes());
            modelValues.lycoris = isEmpty(row.lycoris) ? [] : row.lycoris.map((lycoris) => lycoris.getAttributes());
            modelValues.inversion = isEmpty(row.inversion) ? [] : row.inversion.map((inversion) => inversion.model);
            modelValues.vae = isEmpty(row.vae) ? null : row.vae[0].name;

            if (!isEmpty(row.refiner)) {
                modelValues.refiner = row.refiner[0].model;
                modelValues.refiner_size = row.refiner[0].size;
            }
            
            if (!isEmpty(row.inpainter)) {
                modelValues.inpainter = row.inpainter[0].model;
                modelValues.inpainter_size = row.inpainter[0].size;
            }

            if (!isEmpty(row.config)) {
                for (let defaultConfig of row.config) {
                    modelValues[defaultConfig.configuration_key] = defaultConfig.configuration_value;
                }
            }

            if (!isEmpty(row.scheduler)) {
                for (let scheduler of row.scheduler) {
                    if (scheduler.context === "multi_diffusion") {
                        modelValues.multi_scheduler = scheduler.name;
                    } else {
                        modelValues.scheduler = scheduler.name;
                    }
                }
            }

            let modelForm = new ModelForm(this.config, deepClone(modelValues)),
                modelWindow;
            
            modelForm.onChange(async (updatedValues) => {
                if (!isEmpty(modelForm.values.refiner)) {
                    modelForm.addClass("show-refiner");
                } else {
                    modelForm.removeClass("show-refiner");
                }
                if (!isEmpty(modelForm.values.inpainter)) {
                    modelForm.addClass("show-inpainter");
                } else {
                    modelForm.removeClass("show-inpainter");
                }
            });
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
                modelForm.enable();
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
        defaultEngineSize = this.application.config.model.invocation.defaultEngineSize;
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
    MultiInversionInputView,
    EngineSizeInputView,
    RefinerEngineSizeInputView,
    InpainterEngineSizeInputView,
    VAEInputView,
    SchedulerInputView,
    MultiDiffusionSchedulerInputView
};
