/** @module view/nodes/image-editor */
import { NodeView, OptionsNodeView } from "./base.mjs";
import { ImageView, BackgroundImageView } from "../image.mjs";
import { NodeEditorView } from "./editor.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { SimpleNotification } from "../../common/notify.mjs";
import { FormView } from "../forms/base.mjs";
import { ToolbarView } from "../menu.mjs";
import { ScribbleView } from "../scribble.mjs";
import { 
    SelectInputView,
    CheckboxInputView,
    FloatInputView,
    NumberInputView,
    TextInputView
} from "../forms/input.mjs";

const E = new ElementBuilder();

/**
 * Filter out empty keys
 */
const filterEmpty = (obj) => {
    let values = {};
    for (let key in obj) {
        if (!isEmpty(obj[key])) {
            values[key] = obj[key];
        }
    }
    return values;
};

class InvocationToolbarView extends ToolbarView {
    constructor(invocationNode) {
        super(invocationNode.config);
        this.invocationNode = invocationNode;
    }

    async onMouseEnter(e) {
        this.invocationNode.toolbarEntered();
    }

    async onMouseLeave(e) {
        this.invocationNode.toolbarLeft();
    }

    async build() {
        let node = await super.build();
        node.on("mouseenter", (e) => this.onMouseEnter(e));
        node.on("mouseleave", (e) => this.onMouseLeave(e));
        return node;
    }
};

/**
 * Create a small extension of the ImageView to change the class name for CSS.
 */
class CurrentInvocationImageView extends ImageView {
    /**
     * Constructed by the editor, pass reference so we can call other functions
     */
    constructor(editor) {
        super(editor.config);
        this.editor = editor;
    }

    /**
     * @var string The class name to apply to the image node
     */
    static className = "current-invocation-image-view";

    /**
     * @var int The number of milliseconds to wait after leaving the image to hide tools
     */
    static hideTime = 250;

    /**
     * Gets the toolbar node, building if needed
     */
    async getTools() {
        if (isEmpty(this.toolbar)) {
            this.toolbar = new InvocationToolbarView(this);
            
            if (!!navigator.clipboard && typeof ClipboardItem === "function") {
                this.copyImage = await this.toolbar.addItem("Copy to Clipboard", "fa-solid fa-clipboard");
                this.copyImage.onClick(() => this.copyToClipboard());
            }

            this.popoutImage = await this.toolbar.addItem("Popout Image", "fa-solid fa-arrow-up-right-from-square");
            this.popoutImage.onClick(() => this.sendToWindow());

            this.saveImage = await this.toolbar.addItem("Save As", "fa-solid fa-floppy-disk");
            this.saveImage.onClick(() => this.saveToDisk());

            this.editImage = await this.toolbar.addItem("Edit Image", "fa-solid fa-pen-to-square");
            this.editImage.onClick(() => this.sendToCanvas());
        }
        return this.toolbar;
    }

    /**
     * Triggers the copy to clipboard
     * Chromium only as of 2023-06-21
     */
    async copyToClipboard() {
        navigator.clipboard.write([
            new ClipboardItem({
                "image/png": await this.getBlob()
            })
        ]);
        SimpleNotification.notify("Copied to clipboard!", 2000);
    }

    /**
     * Saves the image to disk
     * Asks for a filename first
     */
    async saveToDisk() {
        this.editor.application.saveBlobAs("Save Image", await this.getBlob(), ".png");
    }

    /**
     * Sends the image to a new canvas
     * Asks for details regarding additional state when clicked
     */
    async sendToCanvas() {
        this.editor.application.initializeStateFromImage(await this.getImageAsDataURL());
    }

    /**
     * Opens the image in a new window
     */
    async sendToWindow() {
        window.open(this.src);
    }

    /**
     * The callback when the toolbar has been entered
     */
    async toolbarEntered() {
        this.stopHideTimer();
    }

    /**
     * The callback when the toolbar has been left
     */
    async toolbarLeft() {
        this.startHideTimer();
    }

    /**
     * Stops the timeout that will hide tools
     */
    stopHideTimer() {
        clearTimeout(this.timer);
    }

    /**
     * Start the timeout that will hide tools
     */
    startHideTimer() {
        this.timer = setTimeout(async () => {
            let release = await this.lock.acquire();
            let toolbar = await this.getTools();
            this.node.element.parentElement.removeChild(await toolbar.render());
            release();
        }, this.constructor.hideTime);
    }

    /**
     * The callback for MouseEnter
     */
    async onMouseEnter(e) {
        this.stopHideTimer();
        let release = await this.lock.acquire();
        let toolbar = await this.getTools();
        this.node.element.parentElement.appendChild(await toolbar.render());
        release();
    }

    /**
     * The callback for MouesLeave
     */
    async onMouseLeave(e) {
        this.startHideTimer();
    }

    /**
     * On build, bind mouseenter to show tools
     */
    async build() {
        let node = await super.build();
        node.on("mouseenter", (e) => this.onMouseEnter(e));
        node.on("mouseleave", (e) => this.onMouseLeave(e));
        return node;
    }
};

/**
 * These are the ControlNet options
 */
class ImageEditorImageControlNetInputView extends SelectInputView {
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
        "mlsd": "Mobile Line Segment Detection (MLSD)"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "The ControlNet to use depends on your input image. Unless otherwise specified, your input image will be processed through the appropriate algorithm for this ControlNet prior to diffusion.<br /><strong>Canny Edge</strong>: This network is trained on images and the edges of that image after having run through Canny Edge detection.<br /><strong>HED</strong>: Short for Holistically-Nested Edge Detection, this edge-detection algorithm is best used when the input image is too blurry or too noisy for Canny Edge detection.<br /><strong>MLSD</strong>: Short for Mobile Line Segment Detection, this edge-detection algorithm searches only for straight lines, and is best used for geometric or architectural images.";
};


/**
 * The fit options
 */
class ImageEditorImageFitInputView extends SelectInputView {
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
class ImageEditorImageAnchorInputView extends SelectInputView {
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

class ImageEditorBaseOptionsFormView extends FormView {
    /**
     * @var object The fieldsets of the options form for image mode.
     */
    static fieldSets = {
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
        "Tweaks": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": 7.5,
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
                    "value": 50,
                    "tooltip": "How many steps to take during primary inference, larger values take longer to process."
                }
            }
        },
        "Other": {
            "scaleToModelSize": {
                "label": "Scale to Model Size",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "When this node has any dimension smaller than the size of the configured model, scale it up so it's smallest dimension is the same size as the model, then scale it down after diffusion.<br />This generally improves image quality in rectangular shapes, but can also result in ghosting and increased processing time.<br />This will have no effect if your node is larger than the model size in all dimensions.<br />If unchecked and your node is smaller than the model size, TensorRT will be disabled for this node."
                },
            },
            "removeBackground": {
                "label": "Remove Background",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "After diffusion, run the resulting image though an AI background removal algorithm. This can improve image consistency when using multiple nodes."
                }
            }
        },
    };
    
    /**
     * @var bool Never show submit button
     */
    static autoSubmit = true;

    /**
     * @var string An additional classname for this form
     */
    static className = "options-form-view";
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
                "class": ImageEditorImageFitInputView
            },
            "anchor": {
                "label": "Image Anchor",
                "class": ImageEditorImageAnchorInputView
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
                    "value": true,
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
        "Tweaks": {
            "guidanceScale": {
                "label": "Guidance Scale",
                "class": FloatInputView,
                "config": {
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "value": 7.5,
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
                    "value": 50,
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
                "class": ImageEditorImageControlNetInputView
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
        }
    };

    /**
     * @var object The conditions for display of some inputs.
     */
    static fieldSetConditions = {
        "Prompts": (values) => values.infer || values.inpaint || values.control,
        "Tweaks": (values) => values.infer || values.inpaint || values.control,
        "Inference": (values) => values.infer,
        "Control": (values) => values.control,
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

/**
 * Nodes on the Image Editor use multiples of 8 instead of 10
 */
class ImageEditorNodeView extends NodeView {
    /**
     * @var bool Enable header flipping
     */
    static canFlipHeader = true;

    /**
     * @var int The minimum height, much smaller than normal minimum.
     */
    static minHeight = 32;
    
    /**
     * @var int The minimum width, much smaller than normal minimum.
     */
    static minWidth = 32;

    /**
     * @var int Change snap size from 10 to 8
     */
    static snapSize = 8;

    /**
     * @var int Change padding from 10 to 8
     */
    static padding = 8;

    /**
     * @var int Change edge handler tolerance from 10 to 8
     */
    static edgeHandlerTolerance = 8;

    /**
     * @var bool All nodes on the image editor try to be as minimalist as possible.
     */
    static hideHeader = true;

    /**
     * @var string Change from 'Close' to 'Remove'
     */
    static closeText = "Remove";
    
    /**
     * @var array<object> The buttons for the node.
     * @see view/nodes/base
     */
    static nodeButtons = {
        anchor: {
            icon: "fa-solid fa-sliders",
            tooltip: "Show/Hide Options",
            callback: function() {
                this.toggleOptions();
            }
        }
    };

    /**
     * @var class The form to use. Each node should have their own.
     */
    static optionsFormView = ImageEditorBaseOptionsFormView;

    /**
     * Can be overridden in the node classes; this is called when their options are changed.
     */
    async updateOptions(values) {
        this.prompt = values.prompt;
        this.negativePrompt = values.negativePrompt;
        this.guidanceScale = values.guidanceScale;
        this.inferenceSteps = values.inferenceSteps;
        this.scaleToModelSize = values.scaleToModelSize;
        this.removeBackground = values.removeBackground;
    }

    /**
     * Shows the options view.
     */
    async toggleOptions() {
        if (isEmpty(this.optionsForm)) {
            this.optionsForm = new this.constructor.optionsFormView(this.config);
            this.optionsForm.onSubmit((values) => this.updateOptions(values));
            let optionsNode = await this.optionsForm.getNode();
            this.optionsForm.setValues(this.getState());
            this.node.find("enfugue-node-contents").append(optionsNode);
        } else if (this.optionsForm.hidden) {
            this.optionsForm.show();
        } else {
            this.optionsForm.hide();
        }
    }

    /**
     * When state is set, send to form
     */
    setState(newState) {
        super.setState({
            name: newState.name,
            x: newState.x - this.constructor.padding,
            y: newState.y - this.constructor.padding,
            h: newState.h + (this.constructor.padding * 2),
            w: newState.w + (this.constructor.padding * 2)
        });
        
        this.updateOptions(newState);
        
        if (!isEmpty(this.optionsForm)) {
            this.optionsForm.setValues(newState);
        }
    }

    /**
     * Gets the base state and appends form values.
     */
    getState() {
        let state = super.getState();
        state.prompt = this.prompt || null;
        state.negativePrompt = this.negativePrompt || null;
        state.guidanceScale = this.guidanceScale || 7.5;
        state.inferenceSteps = this.inferenceSteps || 50;
        state.removeBackground = this.removeBackground || false;
        state.scaleToModelSize = this.scaleToModelSize || true;
        return state;
    }
};

/**
 * The PromptNode just allows for regions to have different prompts.
 */
class ImageEditorPromptNodeView extends ImageEditorNodeView {
    /**
     * @var bool Disable header hiding
     */
    static hideHeader = false;
    
    /**
     * @var bool Disable header flipping
     */
    static canFlipHeader = false;

    /**
     * @var string The default node name
     */
    static nodeName = "Prompt";

    /**
     * @var object Remove other buttons
     */
    static nodeButtons = {};

    /**
     * @var string The classname of the node
     */
    static className = "image-editor-prompt-node-view";

    /**
     * Intercept the constructor to set the contents to the options.
     */
    constructor(editor, name, content, left, top, width, height) {
        let realContent = new ImageEditorBaseOptionsFormView(editor.config);
        super(editor, name, realContent, left, top, width, height);
        realContent.onSubmit((values) => this.updateOptions(values));
    }

    /**
     * Gets state from the content
     */
    getState() {
        let state = super.getState();
        state = {...state, ...this.content.values};
        return state;
    }

    /**
     * Set the state on the content
     */
    async setState(newState) {
        await super.setState(newState);
        await this.content.getNode(); // Wait for first build
        await this.content.setValues(newState);
    }
};

/**
 * The ScribbleNodeView allows for drawing and using ControlNet scribble to inference.
 */
class ImageEditorScribbleNodeView extends ImageEditorNodeView {
    /**
     * @var string The icon for changing the cursor to a square.
     */
    static pencilSquareIcon = "fa-regular fa-square";

    /**
     * @var string The tooltip for changing the cursor to a square
     */
    static pencilSquareTooltip = "Change Pencil Shape to Square";
    
    /**
     * @var string The icon for changing the cursor to a circle.
     */
    static pencilCircleIcon = "fa-regular fa-circle";

    /**
     * @var string The tooltip for changing the cursor to a circle
     */
    static pencilCircleTooltip = "Change Pencil Shape to Circle";
    
    /**
     * @var string The icon for changing to eraser mode
     */
    static eraserIcon = "fa-solid fa-eraser";

    /**
     * @var string The tooltip for changing to eraser mode
     */
    static eraserTooltip = "Switch to Eraser (Hold <code>alt</code> To Quick-Toggle)";
    
    /**
     * @var string The icon for changing to pencil mode
     */
    static pencilIcon = "fa-solid fa-pencil";

    /**
     * @var string The tooltip for changing to pencil mode
     */
    static pencilTooltip = "Switch Back to Pencil";

    /**
     * @var object Buttons to control the scribble. Shortcuts are registered on the view itself.
     */
    static nodeButtons = {
        ...ImageEditorNodeView.nodeButtons,
        ...{
            "shape": {
                "icon": ImageEditorScribbleNodeView.pencilSquareIcon,
                "tooltip": ImageEditorScribbleNodeView.pencilSquareTooltip,
                "callback": function() {
                    this.togglePencilShape();
                }
            },
            "erase": {
                "icon": ImageEditorScribbleNodeView.eraserIcon,
                "tooltip": ImageEditorScribbleNodeView.eraserTooltip,
                "callback": function() {
                    this.toggleEraser();
                }
            },
            "clear": {
                "icon": "fa-solid fa-delete-left",
                "tooltip": "Clear the entire canvas.",
                "callback": function() {
                    this.scribbleView.clearMemory();
                }
            },
            "increase": {
                "icon": "fa-solid fa-plus",
                "tooltip": "Increase Pencil Size",
                "callback": function() {
                    this.scribbleView.increaseSize();
                }
            },
            "decrease": {
                "icon": "fa-solid fa-minus",
                "tooltip": "Decrease Pencil Size",
                "callback": function() {
                    this.scribbleView.decreaseSize();
                }
            }
        }
    };

    /**
     * Toggle the shape of the scribble pencil
     */
    togglePencilShape() {
        let currentShape = this.scribbleView.shape,
            button = this.node.find(".node-button-shape"),
            icon = button.find("i");

        if (currentShape === "circle") {
            this.scribbleView.shape = "square";
            button.data("tooltip", this.constructor.pencilCircleTooltip);
            icon.class(this.constructor.pencilCircleIcon);
        } else {
            this.scribbleView.shape = "circle";
            button.data("tooltip", this.constructor.pencilSquareTooltip);
            icon.class(this.constructor.pencilSquareIcon);
        }
    };
    
    /**
     * Toggles erase mode
     */
    toggleEraser() {
        let currentEraser = this.scribbleView.isEraser === true,
            button = this.node.find(".node-button-erase"),
            icon = button.find("i");

        if (currentEraser) {
            this.scribbleView.isEraser = false;
            button.data("tooltip", this.constructor.eraserTooltip);
            icon.class(this.constructor.eraserIcon);
        } else {
            this.scribbleView.isEraser = true;
            this.scribbleView.shape = "circle";
            button.data("tooltip", this.constructor.pencilTooltip);
            icon.class(this.constructor.pencilIcon);
        }
    };

    /**
     * Intercept the constructor and add ScribbleView
     */
    constructor(editor, name, content, left, top, width, height) {
        let scribbleView = new ScribbleView(editor.config, width, height);
        super(editor, name, scribbleView, left, top, width, height);
        this.scribbleView = this.content; // Set this so the ScribbleView code can be shared with ImageView
    }

    /**
     * When resized, pass to the scribble node to resize itself too
     */
    async resized() {
        super.resized();
        this.scribbleView.resizeCanvas(
            this.visibleWidth - this.constructor.padding * 2,
            this.visibleHeight - this.constructor.padding * 2
        );
    }
    
    /**
     * Override getState to add the scribble and form state
     */
    getState() {
        let state = super.getState();
        state.src = this.scribbleView.src;
        return state;
    }

    /**
     * Override setState to additionally re-build the canvas
     */
    setState(newState) {
        super.setState(newState);
        if (!isEmpty(newState.src)) {
            let imageInstance = new Image();
            imageInstance.onload = () => this.scribbleView.setMemory(imageInstance);
            imageInstance.src = newState.src;
        } else {
            this.scribbleView.clearMemory();
        }
    }
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
        state.processControlImage = this.processControlImage || true;
        state.conditioningScale = this.conditioningScale || 1.0;
        state.removeBackground = this.removeBackground || false;
        state.scaleToModelSize = this.scaleToModelSize || true;
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
            "inferenceSteps": 50,
            "guidanceScale": 7.5,
            "strength": 0.8,
            "processControlImage": true,
            "conditioningScale": 1.0,
            "removeBackground": false,
            "scaleToModelSize": true,
        };
    }
};

/**
 * The ImageEditorView references classes.
 * It also manages to invocation view to view results.
 */
class ImageEditorView extends NodeEditorView {
    /**
     * Image editor contains ref to app.
     */
    constructor(application) {
        super(application.config, window.innerWidth-300, window.innerHeight-70);
        this.application = application;
    }

    /**
     * @var int The width of the canvas (can be changed)
     */
    static canvasWidth = 512;

    /**
     * @var int The height of the canvas (can be changed)
     */
    static canvasHeight = 512;

    /**
     * @var bool Center the editor
     */
    static centered = true;

    /**
     * @var string Add the classname for CSS
     */
    static className = 'image-editor';

    /**
     * @var int Increase the maximum zoom by a lot
     */
    static maximumZoom = 10;

    /**
     * @var array<class> The node classes for state set/get
     */
    static nodeClasses = [
        ImageEditorScribbleNodeView,
        ImageEditorImageNodeView,
        ImageEditorPromptNodeView
    ];

    /**
     * Removes the current invocation from the canvas view.
     */
    hideCurrentInvocation() {
        this.currentInvocation.hide();
        this.removeClass("has-image");
        if (!isEmpty(this.configuredWidth) && !isEmpty(this.configuredHeight)) {
            this.width = this.configuredWidth;
            this.height = this.configuredHeight;
            this.configuredHeight = null;
            this.configuredWidth = null;
        }
    }

    /**
     * Sets a current invocation on the canvas view.
     * @param string $href The image source.
     */
    async setCurrentInvocationImage(href) {
        this.currentInvocation.setImage(href);
        await this.currentInvocation.waitForLoad();
        if (this.currentInvocation.width != this.width || this.currentInvocation.height != this.height) {
            if (isEmpty(this.configuredWidth)) {
                this.configuredWidth = this.width;
            }
            if (isEmpty(this.configuredHeight)) {
                this.configuredHeight = this.height;
            }
            this.width = this.currentInvocation.width;
            this.height = this.currentInvocation.height;
        }
        this.currentInvocation.show();
        this.addClass("has-image");
    }

    /**
     * Gets the next unoccupied [x, y]
     */
    getNextNodePoint() {
        let nodeX = this.nodes.map((node) => node.left + ImageEditorNodeView.padding),
            nodeY = this.nodes.map((node) => node.top + ImageEditorNodeView.padding),
            [x, y] = [0, 0];
        
        while (nodeX.indexOf(x) !== -1) x += ImageEditorNodeView.snapSize;
        while (nodeY.indexOf(y) !== -1) y += ImageEditorNodeView.snapSize;
        return [x, y];
    }

    /**
     * This is a shorthand helper functinon for adding an image node.
     * @param string $imageSource The source of the image - likely a data URL.
     * @return NodeView The added view.
     */
    async addImageNode(imageSource, imageName = "Image") {
        let imageView = new ImageView(this.config, imageSource),
            [x, y] = this.getNextNodePoint();
        await imageView.waitForLoad();
        return await this.addNode(
            ImageEditorImageNodeView,
            imageName,
            imageView,
            x,
            y,
            imageView.width,
            imageView.height
        );
    }

    /**
     * This is a shorthand helper for adding a scribble node.
     * @return NodeView The added view
     */
    async addScribbleNode(scribbleName = "Scribble") {
        let [x, y] = this.getNextNodePoint();
        return await this.addNode(
            ImageEditorScribbleNodeView,
            scribbleName,
            null,
            x,
            y,
            256,
            256
        );
    }
    
    /**
     * This is a shorthand helper for adding a prompt node.
     * @return NodeView The added view.
     */
    async addPromptNode(promptName = "Prompt") {
        let [x, y] = this.getNextNodePoint();
        return await this.addNode(
            ImageEditorPromptNodeView,
            promptName,
            null,
            x,
            y,
            256,
            256
        );
    }

    /**
     * Builds the DOMElement
     */
    async build() {
        let node = await super.build(),
            grid = E.createElement("enfugue-image-editor-grid");
        this.currentInvocation = new CurrentInvocationImageView(this);
        this.currentInvocation.hide();
        node.find("enfugue-node-canvas").append(grid, await this.currentInvocation.getNode());
        return node;
    }

    /**
     * Gets base state when initializing from an image
     */
    static getNodeDataForImage(image) {
        let baseState = {
            "x": 0,
            "y": 0,
            "w": image.width,
            "h": image.height,
            "src": image.src,
            "name": "Initial Image"
        };
        return {...baseState, ...ImageEditorImageNodeView.getDefaultState()};        
    }
}

export { 
    ImageEditorView, 
    ImageEditorNodeView,
    ImageEditorImageNodeView,
    ImageEditorScribbleNodeView
};
