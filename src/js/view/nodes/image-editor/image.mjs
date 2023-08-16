/** @module view/nodes/image-editor/image-node.mjs */
import { isEmpty } from "../../../base/helpers.mjs";
import { ImageView, BackgroundImageView } from "../../image.mjs";
import { ImageEditorScribbleNodeView } from "./scribble.mjs";
import { FormView } from "../../forms/base.mjs";
import {
    SelectInputView,
    CheckboxInputView,
    TextInputView,
    NumberInputView,
    FloatInputView
} from "../../forms/input.mjs";

/**
 * This input allows the user to specify what colors an image is, so we can determine
 * if we need to invert them on the backend.
 */
class ImageColorSpaceInputView extends SelectInputView {
    /**
     * @var object Only one option
     */
    static defaultOptions = {
        "invert": "White on Black"
    };

    /**
     * @var string The default option is to invert
     */
    static defaultValue = "invert";

    /**
     * @var string The empty option text
     */
    static placeholder = "Black on White";

    /**
     * @var bool Always show empty
     */
    static allowEmpty = true;
}

/**
 * These are the ControlNet options
 */
class ControlNetInputView extends SelectInputView {
    /**
     * @var string Set the default to the easiest and fastest
     */
    static defaultValue = "canny";

    /**
     * @var object The options allowed.
     */
    static defaultOptions = {
        "canny": "Canny Edge Detection",
        "hed": "Holistically-nested Edge Detection (HED)",
        "pidi": "Soft Edge Detection (PIDI)",
        "mlsd": "Mobile Line Segment Detection (MLSD)",
        "line": "Line Art",
        "anime": "Anime Line Art",
        "scribble": "Scribble",
        "depth": "Depth Detection (MiDaS)",
        "normal": "Normal Detection (Estimate)",
        "pose": "Pose Detection (OpenPose)"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "The ControlNet to use depends on your input image. Unless otherwise specified, your input image will be processed through the appropriate algorithm for this ControlNet prior to diffusion.<br />" +
        "<strong>Canny Edge</strong>: This network is trained on images and the edges of that image after having run through Canny Edge detection.<br />" +
        "<strong>HED</strong>: Short for Holistically-Nested Edge Detection, this edge-detection algorithm is best used when the input image is too blurry or too noisy for Canny Edge detection.<br />" +
        "<strong>Soft Edge Detection</strong>: Using a Pixel Difference Network, this edge-detection algorithm can be used in a wide array of applications.<br />" +
        "<strong>MLSD</strong>: Short for Mobile Line Segment Detection, this edge-detection algorithm searches only for straight lines, and is best used for geometric or architectural images.<br />" +
        "<strong>Line Art</strong>: This model is capable of rendering images to line art drawings. The controlnet was trained on the model output, this provides a great way to provide your own hand-drawn pieces as well as another means of edge detection.<br />" +
        "<strong>Anime Line Art</strong>: This is similar to the above, but focusing specifically on anime style.<br />" +
        "<strong>Scribble</strong>: This ControlNet was trained on a variant of the HED edge-detection algorithm, and is good for hand-drawn scribbles with thick, variable lines.<br />" +
        "<strong>Depth</strong>: This uses Intel's MiDaS model to estimate monocular depth from a single image. This uses a greyscale image showing the distance from the camera to any given object.<br />" +
        "<strong>Normal</strong>: Normal maps are similar to depth maps, but instead of using a greyscale depth, three sets of distance data is encoded into red, green and blue channels.<br />" +
        "<strong>OpenPose</strong>: This AI model from the Carnegie Mellon University's Perceptual Computing Lab detects human limb, face and digit poses from an image. Using this data, you can generate different people in the same pose.";
};


/**
 * The fit options
 */
class ImageFitInputView extends SelectInputView {
    /**
     * @var string The default value is the first
     */
    static defaultValue = "actual";

    /**
     * @var object The Options allowed
     */
    static defaultOptions = {
        "actual": "Actual Size",
        "stretch": "Stretch",
        "contain": "Contain",
        "cover": "Cover"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "Fit this image within it's container.<br /><strong>Actual</strong>: Do not manipulate the dimensions of the image, anchor it as specified in the frame.<br /><strong>Stretch</strong>: Force the image to fit within the frame, regardles sof original dimensions. When using this mode, anchor has no effect.<br /><strong>Contain</strong>: Scale the image so that it's largest dimension is contained within the frames bounds, adding negative space to fill the rest of the frame.<br /><strong>Cover</strong>: Scale the image so that it's smallest dimension is contained within the frame bounds, cropping the rest of the image as needed.";
};

/**
 * The anchor options
 */
class ImageAnchorInputView extends SelectInputView {
    /**
     * @var string The default value is the first
     */
    static defaultValue = "top-left";

    /**
     * @var object The Options allowed
     */
    static defaultOptions = {
        "top-left": "Top Left",
        "top-center": "Top Center",
        "top-right": "Top Right",
        "center-left": "Center Left",
        "center-center": "Center Center",
        "center-right": "Center Right",
        "bottom-left": "Bottom Left",
        "bottom-center": "Bottom Center",
        "bottom-right": "Bottom Right",
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "When the size of the frame and the size of the image do not match, this will control where the image is placed. View the 'fit' field for more options to fit images in the frame.";
};

/**
 * This form combines all image options.
 */
class ImageEditorImageNodeOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
        "Base": {
            "fit": {
                "label": "Image Fit",
                "class": ImageFitInputView
            },
            "anchor": {
                "label": "Image Anchor",
                "class": ImageAnchorInputView
            },
            "inpaint": {
                "label": "Use for Inpainting",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, you will be able to paint where on the image you wish for the AI to fill in details. Any gaps in the frame or transparency in the image will also be filled."
                }
            },
            "infer": {
                "label": "Use for Inference",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image as input for primary diffusion. Inpainting will be performed first, filling any painted sections as well as gaps in the frame and transparency in the image."
                }
            },
            "control": {
                "label": "Use for Control",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, use this image as input for ControlNet. Inpainting will be performed first, filling any painted sections as well as gaps in the frame and transparency in the image.<br />Unless otherwise specified, your image will be processed using the appropriate algorithm for the chosen ControlNet."
                }
            },
        },
        "Other": {
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
        "Prompts": {
            "prompt": {
                "label": "Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global prompt will be used."
                }
            },
            "negativePrompt": {
                "label": "Negative Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global negative prompt will be used."
                }
            },
        },
        "Secondary Prompts": {
            "prompt2": {
                "label": "Secondary Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in this frame. When left blank, the global secondary prompt will be used. Secondary prompts are input into the secondary text encoder when using SDXL. When not using SDXL, secondary prompts will be merged with primary prompts."
                }
            },
            "negativePrompt2": {
                "label": "Secondary Negative Prompt",
                "class": TextInputView,
                "config": {
                    "tooltip": "This prompt will control what is in not this frame. When left blank, the global secondary negative prompt will be used. Secondary negative prompts are input ito the secondary text encoder when using SDXL. When not using SDXL, secondary prompts will be merged with primary prompts."
                }
            },
        },
        "Tweaks": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": null,
                    "tooltip": "How closely to follow the text prompt; high values result in high-contrast images closely adhering to your text, low values result in low-contrast images with more randomness."
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
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process."
                }
            }
        },
        "Inference": {
            "strength": {
                "label": "Denoising Strength",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 0.8,
                    "config": "How much of the input image to replace with new information. A value of 1.0 represents total input image destruction, and 0.0 represents no image modifications being made."
                }
            }
        },
        "Control": {
            "controlnet": {
                "label": "ControlNet",
                "class": ControlNetInputView
            },
            "conditioningScale": {
                "label": "Conditioning Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                    "config": "How closely to follow ControlNet's influence."
                }
            },
            "processControlImage": {
                "label": "Process Image for ControlNet",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "When checked, the image will be processed through the appropriate edge detection algorithm for the ControlNet. Only uncheck this if your image has already been processed through edge detection."
                }
            }
        },
        "Color Space": {
            "colorSpace": {
                "label": "Input Image Color Space",
                "class": ImageColorSpaceInputView
            }
        }
    };

    /**
     * @var object The conditions for display of some inputs.
     */
    static fieldSetConditions = {
        "Prompts": (values) => values.infer || values.inpaint || values.control,
        "Secondary Prompts": (values) => values.infer || values.inpaint || values.control,
        "Tweaks": (values) => values.infer || values.inpaint || values.control,
        "Inference": (values) => values.infer,
        "Control": (values) => values.control,
        "Color Space": (values) => values.control && 
            ["mlsd", "hed", "pidi", "scribble", "line", "anime"].indexOf(values.controlnet) !== -1 &&
            values.processControlImage === false
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
    static collapseFieldSets = ["Secondary Prompts", "Tweaks"];
};

/**
 * When pasting images on the image editor, allow a few fit options
 */
class ImageEditorImageNodeView extends ImageEditorScribbleNodeView {
    /**
     * @var array<string> All fit modes.
     */
    static allFitModes = ["actual", "stretch", "cover", "contain"];

    /**
     * @var array<string> All anchor modes.
     */
    static allAnchorModes = [
        "top-left", "top-center", "top-right",
        "center-left", "center-center", "center-right",
        "bottom-left", "bottom-center", "bottom-right"
    ];

    /**
     * @var array<string> The node buttons that pertain to scribble.
     */
    static scribbleButtons = [
        "erase", "shape", "clear", "increase", "decrease"
    ];
    
    /**
     * @var string Add the classname for CSS
     */
    static className = 'image-editor-image-node-view';

    /**
     * @var object Buttons to control the scribble. Shortcuts are registered on the view itself.
     */
    static nodeButtons = {
        ...ImageEditorScribbleNodeView.nodeButtons,
        ...{
            "mirror-x": {
                "icon": "fa-solid fa-left-right",
                "tooltip": "Mirror the image horizontally.",
                "callback": function() {
                    this.mirrorHorizontally();
                }
            },
            "mirror-y": {
                "icon": "fa-solid fa-up-down",
                "tooltip": "Mirror the image vertically.",
                "callback": function() {
                    this.mirrorVertically();
                }
            },
            "rotate-clockwise": {
                "icon": "fa-solid fa-rotate-right",
                "tooltip": "Rotate the image clockwise by 90 degrees.",
                "callback": function() {
                    this.rotateClockwise();
                }
            },
            "rotate-counter-clockwise": {
                "icon": "fa-solid fa-rotate-left",
                "tooltip": "Rotate the image counter-clockwise by 90 degrees.",
                "callback": function() {
                    this.rotateCounterClockwise();
                }
            }
        }
    };

    /**
     * @var class The form for this node.
     */
    static optionsFormView = ImageEditorImageNodeOptionsFormView;

    /**
     * Intercept the constructor to set the contents to use background image instead of base image.
     */
    constructor(editor, name, content, left, top, width, height) {
        if (content instanceof ImageView) {
            content = new BackgroundImageView(content.config, content.src);
        }
        super(editor, name, null, left, top, width, height);
        this.scribbleView = this.content;
        this.content = content;
        // Hide scribbble view 
        this.scribbleView.hide();
    }

    /**
     * Updates the options after a user makes a change.
     */
    async updateOptions(newOptions) {
        super.updateOptions(newOptions);

        // Reflected in DOM
        this.updateFit(newOptions.fit);
        this.updateAnchor(newOptions.anchor);
        
        // Flags
        this.infer = newOptions.infer;
        this.control = newOptions.control;
        this.inpaint = newOptions.inpaint;

        // Conditional inputs
        this.strength = newOptions.strength;
        this.controlnet = newOptions.controlnet;
        this.conditioningScale = newOptions.conditioningScale;
        this.processControlImage = newOptions.processControlImage;

        // Update scribble view if inpainting
        if (this.node !== undefined) {
            let nodeHeader = this.node.find("enfugue-node-header");
            if (this.inpaint) {
                this.scribbleView.show();
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).show();
                }
            } else {
                this.scribbleView.hide();
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).hide();
                }
            }
        }
    };

    /**
     * Updates the image fit
     */
    async updateFit(newFit) {
        this.fit = newFit;
        let content = await this.getContent();
        content.fit = newFit;
        for (let fitMode of this.constructor.allFitModes) {
            content.removeClass(`fit-${fitMode}`);
        }
        if (!isEmpty(newFit)) {
            content.addClass(`fit-${newFit}`);
        }
    };

    /**
     * Updates the image anchor
     */
    async updateAnchor(newAnchor) {
        this.anchor = newAnchor;
        let content = await this.getContent();
        for (let anchorMode of this.constructor.allAnchorModes) {
            content.removeClass(`anchor-${anchorMode}`);
        }
        if (!isEmpty(newAnchor)) {
            content.addClass(`anchor-${newAnchor}`);
        }
    }
    
    /**
     * On build, append the scribble view (hidden) and hide scribble buttons
     */
    async build() {
        let node = await super.build();
        node.find("enfugue-node-contents").append(await this.scribbleView.getNode());
        let nodeHeader = node.find("enfugue-node-header");
        for (let button of this.constructor.scribbleButtons) {
            nodeHeader.find(`.node-button-${button}`).hide();
        }
        return node;
    }

    /**
     * Mirrors the image horizontally
     */
    mirrorHorizontally() {
        return this.content.mirrorHorizontally();
    }
    
    /**
     * Mirrors the image vertically
     */
    mirrorVertically() {
        return this.content.mirrorVertically();
    }
    
    /**
     * Rotates the image clockwise by 90 degrees
     */
    rotateClockwise() {
        return this.content.rotateClockwise();
    }
    
    /**
     * Rotates the image counter-clockwise by 90 degrees
     */
    rotateCounterClockwise() {
        return this.content.rotateCounterClockwise();
    }

    /**
     * Override getState to include the image, fit and anchor
     */
    getState() {
        let state = super.getState();
        state.scribbleSrc = this.scribbleView.src;
        state.src = this.content.src;
        state.anchor = this.anchor || null;
        state.fit = this.fit || null;
        state.infer = this.infer || false;
        state.control = this.control || false;
        state.inpaint = this.inpaint || false;
        state.strength = this.strength || 0.8;
        state.controlnet = this.controlnet || null;
        state.colorSpace = this.colorSpace || "invert";
        state.conditioningScale = this.conditioningScale || 1.0;
        state.processControlImage = this.processControlImage !== false;
        state.removeBackground = this.removeBackground === true;
        state.scaleToModelSize = this.scaleToModelSize === true;
        return state;
    }

    /**
     * Override setState to add the image and scribble
     */
    async setState(newState) {
        await this.setContent(new BackgroundImageView(this.config, newState.src));
        await this.updateAnchor(newState.anchor);
        await this.updateFit(newState.fit);
        if (newState.inpaint) {
            let scribbleImage = new Image();
            scribbleImage.onload = () => {
                this.scribbleView.setMemory(scribbleImage);
                this.scribbleView.show();
                if (this.node !== undefined) {
                    let nodeHeader = this.node.find("enfugue-node-header");
                    for (let button of this.constructor.scribbleButtons) {
                        nodeHeader.find(`.node-button-${button}`).show();
                    }
                }
            };
            scribbleImage.src = newState.scribbleSrc;
        } else {
            this.scribbleView.clearMemory();
            this.scribbleView.hide();
            if (this.node !== undefined) {
                let nodeHeader = this.node.find("enfugue-node-header");
                for (let button of this.constructor.scribbleButtons) {
                    nodeHeader.find(`.node-button-${button}`).hide();
                }
            }
        }
        await super.setState({...newState, ...{"src": null}});
    }

    /**
     * Provide a default state for when we are initializing from an image
     */
    static getDefaultState() {
        return {
            "classname": this.name,
            "inpaint": false,
            "control": false,
            "inpaint": false,
            "inferenceSteps": null,
            "guidanceScale": null,
            "strength": 0.8,
            "processControlImage": true,
            "colorSpace": "invert",
            "conditioningScale": 1.0,
            "removeBackground": false,
            "scaleToModelSize": false,
        };
    }
};

export {
    ControlNetInputView,
    ImageAnchorInputView,
    ImageFitInputView,
    ImageColorSpaceInputView,
    ImageEditorImageNodeView,
    ImageEditorImageNodeOptionsFormView
};
