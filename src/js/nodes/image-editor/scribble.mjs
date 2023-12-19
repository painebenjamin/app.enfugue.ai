/** @module nodes/image-editor/scribble.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { ScribbleView } from "../../view/scribble.mjs";
import { ImageEditorNodeView } from "./base.mjs";

/**
 * The ScribbleNodeView allows for drawing and using ControlNet scribble to inference.
 */
class ImageEditorScribbleNodeView extends ImageEditorNodeView {
    /**
     * @var string Name to display in the menu
     */
    static nodeTypeName = "Scribble";

    /**
     * @var bool hide header
     */
    static hideHeader = true;

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
                "shortcut": "e",
                "callback": function() {
                    this.togglePencilShape();
                }
            },
            "erase": {
                "icon": ImageEditorScribbleNodeView.eraserIcon,
                "tooltip": ImageEditorScribbleNodeView.eraserTooltip,
                "shortcut": "t",
                "callback": function() {
                    this.toggleEraser();
                }
            },
            "clear": {
                "icon": "fa-solid fa-delete-left",
                "tooltip": "Clear the entire canvas",
                "shortcut": "l",
                "callback": function() {
                    this.clearMemory();
                }
            },
            "increase": {
                "icon": "fa-solid fa-plus",
                "tooltip": "Increase Pencil Size",
                "shortcut": "i",
                "callback": function() {
                    this.increaseSize();
                }
            },
            "decrease": {
                "icon": "fa-solid fa-minus",
                "tooltip": "Decrease Pencil Size",
                "shortcut": "d",
                "callback": function() {
                    this.decreaseSize();
                }
            }
        }
    };

    /**
     * Intercept the constructor and add ScribbleView
     */
    constructor(editor, name, content, left, top, width, height) {
        super(
            editor,
            name,
            new ScribbleView(editor.config, width, height),
            left,
            top,
            width,
            height
        );
    }

    /**
     * Wait a tick after calling this then redraw
     */
    async scaleToCanvasSize() {
        await super.scaleToCanvasSize();
        window.requestAnimationFrame(() => {
            this.resized();
        });
    }

    /**
     * Calls clear memory on content
     */
    clearMemory() {
        this.content.clearMemory();
    }

    /**
     * Calls increase size on content
     */
    increaseSize() {
        this.content.increaseSize();
    }

    /**
     * Calls decrease size on content
     */
    decreaseSize() {
        this.content.decreaseSize();
    }

    /**
     * Toggle the shape of the scribble pencil
     */
    togglePencilShape() {
        let currentShape = this.content.shape;

        if (currentShape === "circle") {
            this.content.shape = "square";
            this.buttons.shape.tooltip = this.constructor.pencilCircleTooltip;
            this.buttons.shape.icon = this.constructor.pencilCircleIcon;
        } else {
            this.content.shape = "circle";
            this.buttons.shape.tooltip = this.constructor.pencilSquareTooltip;
            this.buttons.shape.icon = this.constructor.pencilSquareIcon;
        }

        this.rebuildHeaderButtons();
    };
    
    /**
     * Toggles erase mode
     */
    toggleEraser() {
        let currentEraser = this.content.isEraser === true;

        if (currentEraser) {
            this.content.isEraser = false;
            this.buttons.erase.icon = this.constructor.eraserIcon;
            this.buttons.erase.tooltip = this.constructor.eraserTooltip;
        } else {
            this.content.isEraser = true;
            this.buttons.erase.icon = this.constructor.pencilIcon;
            this.buttons.erase.tooltip = this.constructor.pencilTooltip;
        }

        this.rebuildHeaderButtons();
    };

    /**
     * When resized, pass to the scribble node to resize itself too
     */
    async resized() {
        await super.resized();
        this.content.resizeCanvas(
            this.visibleWidth - this.constructor.padding * 2,
            this.visibleHeight - this.constructor.padding * 2
        );
    }
    
    /**
     * Override getState to add the scribble and form state
     */
    getState(includeImages = true) {
        let state = super.getState(includeImages);
        state.src = includeImages ? this.content.src : null;
        return state;
    }

    /**
     * Override setState to additionally re-build the canvas
     */
    setState(newState) {
        super.setState(newState);
        if (!isEmpty(newState.src)) {
            let imageInstance = new Image();
            imageInstance.onload = () => this.content.setMemory(imageInstance);
            imageInstance.src = newState.src;
        } else {
            this.content.clearMemory();
        }
    }
};

export { ImageEditorScribbleNodeView };
