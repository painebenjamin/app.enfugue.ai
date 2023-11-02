/** @module forms/enfugue/image-editor/image */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../../forms/base.mjs";
import {
    PromptInputView,
    FloatInputView,
    NumberInputView,
    CheckboxInputView,
    ImageColorSpaceInputView,
    ControlNetInputView,
    ImageFitInputView,
    ImageAnchorInputView,
    FilterSelectInputView,
    SliderPreciseInputView,
    ControlNetUnitsInputView,
} from "../../../forms/input.mjs";

/**
 * This form combines all image options.
 */
class ImageEditorImageNodeOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
        "Image Fit": {
            "fit": {
                "label": "Image Fit",
                "class": ImageFitInputView
            },
            "anchor": {
                "label": "Image Anchor",
                "class": ImageAnchorInputView
            },
         },
         "Image Roles": {
            "inpaint": {
                "label": "Inpainting",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, you will be able to paint where on the image you wish for the AI to fill in details. Any gaps in the frame or transparency in the image will also be filled."
                }
            },
            "infer": {
                "label": "Initialization (img2img)",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image as input for primary diffusion. Inpainting will be performed first, filling any painted sections as well as gaps in the frame and transparency in the image."
                }
            },
            "imagePrompt": {
                "label": "Prompt (IP Adapter)",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image for Image Prompting. This uses a technique whereby your image is analzyed for descriptors automatically and the 'image prompt' is merged with your real prompt. This can help produce variations of an image without adhering too closely to the original image, and without you having to describe the image yourself."
                }
            }
        },
        "Node": {
            "scaleToModelSize": {
                "label": "Scale to Model Size",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When this has any dimension smaller than the size of the configured model, scale it up so it's smallest dimension is the same size as the model, then scale it down after diffusion.<br />This generally improves image quality in rectangular shapes, but can also result in ghosting and increased processing time.<br />This will have no effect if your node is larger than the model size in all dimensions.<br />If unchecked and your node is smaller than the model size, TensorRT will be disabled for this node."
                },
            },
            "removeBackground": {
                "label": "Remove Background",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Before processing, run this image through an AI background removal process. If you are additionally inpainting, inferencing or using this image for ControlNet, that background will then be filled in within this frame. If you are not, that background will be filled when the overall canvas image is finally painted in."
                }
            }
        },
        "Image Prompt": {
            "imagePromptScale": {
                "label": "Image Prompt Scale",
                "class": FloatInputView,
                "config": {
                    "tooltip": "How much strength to give to the image. A higher strength will reduce the effect of your prompt, and a lower strength will increase the effect of your prompt but reduce the effect of the image.",
                    "min": 0,
                    "step": 0.01,
                    "value": 0.5
                }
            },
            "imagePromptPlus": {
                "label": "Use Fine-Grained Model",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Use this to enable fine-grained feature inspection on the source image. This can improve details in the resulting image, but can also make the overall image less similar.<br /><br />Note that when using multiple source images for image prompting, enabling fine-grained feature inspection on any image enables it for all images."
                }
            },
            "imagePromptFace": {
                "label": "Use Face-Specific Model",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Use this to focus strongly on a face in the input image. This can work very well to copy a face from one image to another in a natural way, instead of needing a separate face-fixing step.<br /><br />Note that at present moment, this feature is only available for Stable Diffusion 1.5 models. This checkbox does nothing for SDXL models."
                }
            }
        },
        "Inference": {
            "strength": {
                "label": "Denoising Strength",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.8,
                    "tooltip": "How much of the input image to replace with new information. A value of 1.0 represents total input image destruction, and 0.0 represents no image modifications being made."
                }
            }
        },
        "Inpainting": {
            "cropInpaint": {
                "label": "Use Cropped Inpainting",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, the image will be cropped to the area you've shaded prior to executing. This will reduce processing time on large images, but can result in losing the composition of the image.",
                    "value": true
                }
            },
            "inpaintFeather": {
                "label": "Cropped Inpaint Feather",
                "class": NumberInputView,
                "config": {
                    "min": 16,
                    "max": 256,
                    "step": 8,
                    "value": 32,
                    "tooltip": "When using cropped inpainting, this is the number of pixels to feather along the edge of the crop in order to help blend in with the rest of the image."
                }
            }
        },
        "ControlNet Units": {
            "controlnetUnits": {
                "class": ControlNetUnitsInputView
            }
        },
        "Global Prompt Overrides": {
            "prompt": {
                "label": "Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global prompt will be used."
                }
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": PromptInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global negative prompt will be used."
                }
            },
        },
        "Global Tweaks Overrides": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": null,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness. When left blank, the global guidance scale will be used."
                }
            },
            "inferenceSteps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 5,
                    "max": 250,
                    "step": 1,
                    "value": null,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process. When left blank, the global inference steps will be used."
                }
            }
        }
    };

    /**
     * @var object The conditions for display of some inputs.
     */
    static fieldSetConditions = {
        "Global Prompt Overrides": (values) => values.infer || values.inpaint,
        "Global Tweaks Overrides": (values) => values.infer || values.inpaint,
        "Inpainting": (values) => values.inpaint,
        "Inference": (values) => values.infer,
        "Image Prompt": (values) => values.imagePrompt
    };

    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "image-options-form-view";

    /**
     * @var array Field sets to collapse
     */
    static collapseFieldSets = ["Global Prompt Overrides", "Global Tweaks Overrides"];

    /**
     * On input change, enable/disable flags
     */
    async inputChanged(fieldName, inputView) {
        if (fieldName === "inpaint") {
            let inference = await this.getInputView("infer");
            if (inputView.getValue()) {
                inference.setValue(true, false);
                inference.disable();
                this.values.infer = true;
                this.evaluateConditions();
            } else {
                inference.enable();
            }
        }
        if (fieldName === "imagePromptPlus") {
            if (inputView.getValue()) {
                this.addClass("prompt-plus");
            } else {
                this.removeClass("prompt-plus");
            }
        }
        return super.inputChanged.call(this, fieldName, inputView);
    }

    /**
     * On set values, check and set classes.
     */
    async setValues() {
        await super.setValues.apply(this, Array.from(arguments));
        if (this.values.imagePromptPlus) {
            this.addClass("prompt-plus");
        } else {
            this.removeClass("prompt-plus");
        }
        let inference = await this.getInputView("infer");
        if (this.values.inpaint) {
            this.values.infer = true;
            inference.setValue(true, false);
            inference.disable();
        } else {
            inference.enable();
        }
    }
};

export { ImageEditorImageNodeOptionsFormView };
