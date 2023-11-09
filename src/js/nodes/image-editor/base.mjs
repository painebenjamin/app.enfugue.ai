/** @module nodes/image-editor/base.mjs */
import { isEmpty } from "../../base/helpers.mjs";
import { NodeView } from "../base.mjs";

/**
 * Nodes on the Image Editor use multiples of 8 instead of 10
 */
class ImageEditorNodeView extends NodeView {
    /**
     * @var string The name to show in the menu
     */
    static nodeTypeName = "Base";

    /**
     * @var bool Enable header flipping
     */
    static canFlipHeader = true;

    /**
     * @var int The minimum height, much smaller than normal minimum.
     */
    static minHeight = 64;
    
    /**
     * @var int The minimum width, much smaller than normal minimum.
     */
    static minWidth = 64;

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
     * @var string Change from 'Close' to 'Remove'
     */
    static closeText = "Remove";
    
    /**
     * @var array<object> The buttons for the node.
     * @see view/nodes/base
     */
    static nodeButtons = {
        "nodeToCanvas": {
            "icon": "fa-solid fa-maximize",
            "tooltip": "Scale to Canvas Size",
            "shortcut": "z",
            "callback": function() {
                this.scaleToCanvasSize();
            }
        },
        "canvasToNode": {
            "icon": "fa-solid fa-minimize",
            "tooltip": "Scale Canvas to Image Size",
            "shortcut": "g",
            "callback": function() {
                this.scaleCanvasToSize();
            }
        }
    };

    /**
     * Gets the size to scale to, can be overridden
     */
    async getCanvasScaleSize() {
        return [
            this.width - this.constructor.padding*2,
            this.height - this.constructor.padding*2
        ];
    }

    /**
     * Scales the image up to the size of the canvas
     */
    async scaleToCanvasSize() {
        this.setDimension(
            -this.constructor.padding,
            -this.constructor.padding,
            this.editor.width+this.constructor.padding*2,
            this.editor.height+this.constructor.padding*2,
            true
        );
    }

    /**
     * Scales the canvas size to this size
     */
    async scaleCanvasToSize() {
        let [scaleWidth, scaleHeight] = await this.getCanvasScaleSize();
        this.editor.setDimension(scaleWidth, scaleHeight, true, true);
        this.setDimension(
            -this.constructor.padding,
            -this.constructor.padding,
            scaleWidth+this.constructor.padding*2,
            scaleHeight+this.constructor.padding*2,
            true
        );
    }
};

export { ImageEditorNodeView };
