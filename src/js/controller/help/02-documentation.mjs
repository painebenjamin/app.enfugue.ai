/** @module controller/help/02-documentation */
import { MenuController } from "../menu.mjs";

/**
 * Opens a window to the wiki page
 */
class DocumentationController extends MenuController {
    /**
     * @var string The help page to link to
     */
    static helpLink = "https://github.com/painebenjamin/app.enfugue.ai/wiki";

    /**
     * @var string The text to display
     */
    static menuName = "Documentation";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-book";

    /**
     * On click, open window
     */
    async onClick() {
        window.open(this.constructor.helpLink);
    }
}

export { DocumentationController as MenuController };
