/** @module controller/system/04-logs */
import { MenuController } from "../menu.mjs";

/**
 * The SystemLogsController allows tailing the log for increased visibility
 */
class SystemLogsController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Engine Logs";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-clipboard-list";

    /**
     * When clicked, show logs window.
     */
    async onClick() {
        this.application.logs.showLogDetails();
    }
};

export { SystemLogsController as MenuController };
