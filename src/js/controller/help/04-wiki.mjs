/** @module controller/help/04-wiki */
import { MenuController } from "../menu.mjs";

/**
 * Opens a window to the wiki page
 */
class WikiController extends MenuController {
    /**
     * @var string The help page to link to
     */
    static helpLink = "https://github.com/painebenjamin/app.enfugue.ai/wiki";

    /**
     * @var string The text to display
     */
    static menuName = "Wiki";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-book";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "w";

    /**
     * On click, open window
     */
    async onClick() {
        window.open(this.constructor.helpLink);
    }
}

export { WikiController as MenuController };
