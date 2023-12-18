/** @module controller/sidebar/03-region-prompt */
import { MenuController } from "../menu.mjs";

/**
 * Adds a region prompt node to the image canvas
 */
class RegionPromptController extends MenuController {
    /**
     * @var string Text to display on hover
     */
    static menuName = "Region Prompt";

    /**
     * @var string CSS classes for icon
     */
    static menuIcon = "fa-solid fa-text-width";

    /**
     * On click, add prompt node to canvas
     */
    async onClick() {
        this.canvas.addPromptNode();
    }
}

export { RegionPromptController as ToolbarController };
