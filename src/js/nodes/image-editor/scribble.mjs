/** @module nodes/image-editor/scribble.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { ScribbleView } from "../../view/scribble.mjs";
import { ImageEditorNodeView } from "./base.mjs";

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

export { ImageEditorScribbleNodeView };
