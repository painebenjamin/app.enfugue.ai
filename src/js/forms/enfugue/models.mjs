/** @module forms/enfugue/models */
import { isEmpty } from "../../base/helpers.mjs";
import { ParentView } from "../../view/base.mjs";
import { TableView, ModelTableView } from "../../view/table.mjs";
import { FormView } from "../base.mjs";
import { 
    StringInputView,
    TextInputView,
    NumberInputView,
    CheckboxInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView,
    ModelMergeModeInputView,
    VaeInputView,
    CheckpointInputView,
    EngineSizeInputView,
    InpainterEngineSizeInputView,
    RefinerEngineSizeInputView,
    SchedulerInputView,
    PromptInputView,
    SliderPreciseInputView,
    FloatInputView,
    MaskTypeInputView
} from "../input.mjs";

/**
 * The model form pulls it all together for making/editing models
 */
class ModelFormView extends FormView {
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
                    "tooltip": "Give your model a name that describes what you want it to do - for example, if you're using a photorealistic model and use phrases related to central framing, bokeh focus and and saturated colors, you could call this configuration &ldquo;Product Photography.&rdquo;"
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
                "class": VaeInputView,
                "label": "VAE"
            },
            "refiner": {
                "class": CheckpointInputView,
                "label": "Refining Checkpoint",
                "config": {
                    "tooltip": "Refiner checkpoints were introduced with SDXL 0.9 - these are checkpoints specifically trained to improve detail, shapes, and generally improve the quality of images generated from the base model. These are optional, and do not need to be specifically-trained refinement checkpoints - you can try mixing and matching checkpoints for different styles, though you may wish to ensure the related checkpoints were trained on the same size images."
                }
            },
            "refiner_vae": {
                "class": VaeInputView,
                "label": "Refining VAE"
            },
            "inpainter": {
                "class": CheckpointInputView,
                "label": "Inpainting Checkpoint",
                "config": {
                    "tooltip": "An inpainting checkpoint if much like a regular Stable Diffusion checkpoint, but it additionally includes the ability to input which parts of the image can be changed and which cannot. This is used when you specifically request an image be inpainted, but is also used in many other situations in Enfugue; such as when you place an image on the canvas that doesn't cover the entire space, or use an image that has transparency in it (either before or after removing it's background.) When you don't select an inpainting checkpoint and request an inpainting operation, one will be created dynamically from the main checkpoint at runtime."
                }
            },
            "inpainter_vae": {
                "class": VaeInputView,
                "label": "Inpainting VAE"
            },
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
                "class": PromptInputView,
                "label": "Prompt",
                "tooltip": "This prompt will be appended to every prompt you make when using this model. Use this field to add trigger words, style or quality phrases that you always want to be included."
            },
            "negative_prompt": {
                "class": PromptInputView,
                "label": "Negative Prompt",
                "tooltip": "This prompt will be appended to every negative prompt you make when using this model. Use this field to add trigger words, style or quality phrases that you always want to be excluded."
            }
        },
        "Defaults": {
            "scheduler": {
                "class": SchedulerInputView,
                "label": "Scheduler"
            },
            "num_inference_steps": {
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
            }
        },
        "Refining Defaults": {
            "refiner_start": {
                "label": "Refiner Start",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.85,
                    "tooltip": "When using a refiner, this will control at what point during image generation we should switch to the refiner.<br /><br />For example, if you are using 40 inference steps and this value is 0.5, 20 steps will be performed on the base pipeline, and 20 steps performed on the refiner pipeline. A value of exactly 0 or 1 will make refining it's own step, and instead you can use the 'refining strength' field to control how strong the refinement is."
                }
            },
            "refiner_strength": {
                "label": "Refiner Denoising Strength",
                "class": FloatInputView,
                "config": {
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "step": 0.01,
                    "value": null,
                    "tooltip": "When using a refiner, this will control how much of the original image is kept, and how much of it is replaced with refined content. A value of 1.0 represents total destruction of the first image. This only applies when using refining as it's own distinct step, e.g., the 'refiner start' value is 0 or 1."
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
            },
            "refiner_prompt": {
                "label": "Refiner Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "The prompt to use during refining. By default, the global prompt is used."
                }
            },
            "refiner_negative_prompt": {
                "label": "Refiner Negative Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "The negative prompt to use during refining. By default, the global prompt is used."
                }
            },
        }
    };

    /**
     * @var array Fieldsets to hide
     */
    static collapseFieldSets = [
        "Adaptations and Modifications",
        "Additional Models",
        "Defaults",
        "Refining Defaults"
    ];

    static fieldSetConditions = {
        "Refining Defaults": (values) => !isEmpty(values.refiner)
    };
};

/**
 * This form allows additional pipeline configuration when using a checkpoint
 */
class AbridgedModelFormView extends ModelFormView {
    /**
     * @var string Custom CSS class
     */
    static className = "model-configuration-form-view";

    /**
     * @var boolean no submit button
     */
    static autoSubmit = true;

    /**
     * @var bool No cancel
     */
    static canCancel = false;

    /**
     * @var boolean No hiding
     */
    static collapseFieldSets = false;

    /**
     * @var object one fieldset describes all inputs
     */
    static fieldSets = {
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
                "label": "Textual Inversion",
                "config": {
                    "tooltip": "Textual Inversion is another kind of fine-tuning that teaches novel concepts to Stable Diffusion in a small number of images, which can be used to positively or negatively affect the impact of various prompts."
                }
            }
        },
        "Additional Models": {
            "vae": {
                "label": "VAE",
                "class": VaeInputView
            },
            "refiner": {
                "label": "Refining Checkpoint",
                "class": CheckpointInputView,
                "config": {
                    "tooltip": "Refining checkpoints were introduced with SDXL 0.9 - these are checkpoints specifically trained to improve detail, shapes, and generally improve the quality of images generated from the base model. These are optional, and do not need to be specifically-trained refinement checkpoints - you can try mixing and matching checkpoints for different styles, though you may wish to ensure the related checkpoints were trained on the same size images."
                }
            },
            "refiner_vae": {
                "label": "Refining VAE",
                "class": VaeInputView
            },
            "inpainter": {
                "label": "Inpainting Checkpoint",
                "class": CheckpointInputView,
                "config": {
                    "tooltip": "An inpainting checkpoint if much like a regular Stable Diffusion checkpoint, but it additionally includes the ability to input which parts of the image can be changed and which cannot. This is used when you specifically request an image be inpainted, but is also used in many other situations in Enfugue; such as when you place an image on the canvas that doesn't cover the entire space, or use an image that has transparency in it (either before or after removing it's background.) When you don't select an inpainting checkpoint and request an inpainting operation, one will be created dynamically from the main checkpoint at runtime."
                }
            },
            "inpainter_vae": {
                "label": "Inpainting VAE",
                "class": VaeInputView
            }
        }
    };
};

/**
 * Allows merging 2-3 models together
 */
class MergeModelsFormView extends FormView {
    /**
     * @var Mode of operation and inputs
     */
    static fieldSets = {
        "Method": {
            "method": {
                "class": ModelMergeModeInputView,
                "config": {
                    "required": true
                }
            }
        },
        "Primary Model": {
            "primary": {
                "class": CheckpointInputView,
                "config": {
                    "required": true
                }
            }
        },
        "Secondary Model": {
            "secondary": {
                "class": CheckpointInputView,
                "config": {
                    "required": true
                }
            }
        },
        "Tertiary Model": {
            "tertiary": {
                "class": CheckpointInputView,
                "config": {
                    "tooltip": "When using <strong>Add Difference</strong>, this model should be the subtrahend of the difference equation. The final output model will be of the formula (a + (b - c))."
                }
            }
        },
        "Alpha": {
            "alpha": {
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 1,
                    "step": 0.001,
                    "value": 0.5,
                    "tooltip": "How much weight to apply between the primary and secondary models. A value of 0 represents full weight given to the primary model, a value of 1 is full weight to the secondary model, and 0.5 is equal weight given to both."
                }
            }
        },
        "Output": {
            "filename": {
                "label": "Checkpoint Name",
                "class": StringInputView,
                "config": {
                    "required": true,
                    "placeholder": "My Output(.safetensors)",
                    "tooltip": "Enter a name for the output checkpoint. It will be saved as-is with a `.safetensors` extension."
                }
            }
        }
    };

    /**
     * @var object Conditions for displaying fields
     */
    static fieldSetConditions = {
        "Tertiary Model": (values) => values.method === "add-difference",
        "Alpha": (values) => values.method === "weighted-sum"
    };
};

export { 
    ModelFormView,
    AbridgedModelFormView,
    MergeModelsFormView
};
