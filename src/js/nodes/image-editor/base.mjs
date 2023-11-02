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
    static nodeButtons = {};
};

export { ImageEditorNodeView };
