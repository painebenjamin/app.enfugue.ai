/** @module forms/enfugue/image-editor/image */
import { isEmpty } from "../../../base/helpers.mjs";
import { FormView } from "../../../forms/base.mjs";
import {
    PromptInputView,
    ButtonInputView,
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
        "Image Modifications": {
            "fit": {
                "label": "Image Fit",
                "class": ImageFitInputView
            },
            "anchor": {
                "label": "Image Anchor",
                "class": ImageAnchorInputView
            },
            "removeBackground": {
                "label": "Remove Background",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Before processing, run this image through an AI background removal process. If you are additionally inpainting, inferencing or using this image for ControlNet, that background will then be filled in within this frame. If you are not, that background will be filled when the overall canvas image is finally painted in."
                }
            }
         },
         "Image Roles": {
            "denoise": {
                "label": "Denoising (Image to Image)",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image as input for primary diffusion. If unchecked and no other roles are selected, this will be treated as a pass-through image to be displayed in it's position as-is."
                }
            },
            "imagePrompt": {
                "label": "Prompt (IP Adapter)",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image for Image Prompting. This uses a technique whereby your image is analyzed for descriptors automatically and the 'image prompt' is merged with your real prompt. This can help produce variations of an image without adhering too closely to the original image, and without you having to describe the image yourself."
                }
            },
            "control": {
                "label": "ControlNet (Canny Edge, Depth, etc.)",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image for ControlNet input. This is a technique where your image is processed in some way prior to being used alongside primary inference to try and guide the diffusion process. Effectively, this will allow you to 'extract' features from your image such as edges or a depth map, and transfer them to a new image."
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
        },
        "ControlNet Units": {
            "controlnetUnits": {
                "class": ControlNetUnitsInputView
            }
        }
    };

    /**
     * @var object The conditions for display of some inputs.
     */
    static fieldSetConditions = {
        "Image Prompt": (values) => values.imagePrompt,
        "ControlNet Units": (values) => values.control
    };

    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "image-options-form-view";
};

export { ImageEditorImageNodeOptionsFormView };
