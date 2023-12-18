/** @module controller/toolbar/02-draw-scribble.mjs */
import { MenuController } from "../menu.mjs";
import { promptFiles, isEmpty } from "../../base/helpers.mjs";

/**
 * Adds a scribble node to the canvas
 */
class DrawScribbleController extends MenuController {
    /**
     * @var string Text to display in the tooltip
     */
    static menuName = "Draw Scribble";

    /**
     * @var string CSS class for the icon
     */
    static menuIcon = "fa-solid fa-pencil";

    /**
     * On click, add scribble node
     */
    async onClick() {
        this.canvas.addScribbleNode();
    }
}

export { DrawScribbleController as ToolbarController };
