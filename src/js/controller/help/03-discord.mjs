/** @module controller/help/03-discord */
import { MenuController } from "../menu.mjs";

/**
 * Opens a window to a discussion page
 */
class DiscordController extends MenuController {
    /**
     * @var string The help page to link to
     */
    static helpLink = "https://discord.gg/fESGhyDKvn";

    /**
     * @var string The text to display
     */
    static menuName = "Discord";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-brands fa-discord";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "d";

    /**
     * On click, open window
     */
    async onClick() {
        window.open(this.constructor.helpLink);
    }
}

export { DiscordController as MenuController };
