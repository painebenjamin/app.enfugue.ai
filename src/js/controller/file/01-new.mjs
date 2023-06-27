/** @module controller/file/01-new */
import { MenuController } from "../menu.mjs";

/**
 * The new file controller will simply reset state to default values
 */
class NewFileController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "New";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-file";

    /**
     * On click, reset state
     */
    async onClick() {
        await this.application.resetState();
    }
}

export { NewFileController as MenuController };
