/** @module controller/help/03-discuss */
import { MenuController } from "../menu.mjs";

/**
 * Opens a window to a discussion page
 */
class DiscussionController extends MenuController {
    /**
     * @var string The help page to link to
     */
    static helpLink = "https://github.com/painebenjamin/app.enfugue.ai/discussions";

    /**
     * @var string The text to display
     */
    static menuName = "Discuss";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-comments";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "c";

    /**
     * On click, open window
     */
    async onClick() {
        window.open(this.constructor.helpLink);
    }
}

export { DiscussionController as MenuController };
