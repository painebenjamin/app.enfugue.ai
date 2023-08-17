/** @module forms/enfugue/models */
import { isEmpty } from "../../base/helpers.mjs";
import { ParentView } from "../../view/base.mjs";
import { TableView, ModelTableView } from "../../view/table.mjs";
import { FormView } from "../base.mjs";
import { 
    StringInputView,
    TextInputView,
    NumberInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView,
    VAEInputView,
    CheckpointInputView,
    EngineSizeInputView,
    InpainterEngineSizeInputView,
    RefinerEngineSizeInputView,
    SchedulerInputView,
    MultiDiffusionSchedulerInputView,
    PromptInputView
} from "../input.mjs";

let defaultEngineSize = 512;

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
            "refiner_vae": {
                "class": VAEInputView,
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
                "class": VAEInputView,
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
                    "max": 16384,
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
                    "max": 16384,
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

    /**
     * Checks values and tries to hide/show inputs
     */
    async checkShowInputs() {
        let refinerVAEInput = await this.getInputView("refiner_vae"),
            inpainterVAEInput = await this.getInputView("inpainter_vae");
        
        if (isEmpty(this.values.refiner)) {
            refinerVAEInput.hideParent();
        } else {
            refinerVAEInput.showParent();
        }

        if (isEmpty(this.values.inpainter)) {
            inpainterVAEInput.hideParent();
        } else {
            inpainterVAEInput.showParent();
        }
    }

    /**
     * Hide/show inputs based on change
     */
    async inputChanged(fieldName, inputView) {
        await super.inputChanged(fieldName, inputView);
        await this.checkShowInputs();
    }

    /**
     * On build, hide VAE inputs
     */
    async getNode() {
        let node = await super.getNode();
        setTimeout(() => this.checkShowInputs(), 250);
        return node;
    }
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
     * @var boolean Start hidden
     */
    static collapseFieldSets = true;

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
                "class": VAEInputView
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
                "class": VAEInputView
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
                "class": VAEInputView
            }
        }
    };
};

export { 
    ModelFormView,
    AbridgedModelFormView
};
