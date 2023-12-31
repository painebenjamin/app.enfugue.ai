/** @module controller/models/03-new */
import { MenuController } from "../menu.mjs";

/**
 * Shows the 'new model' form
 */
class NewModelController extends MenuController {
    /**
     * @var string The text in the UI
     */
    static menuName = "New Configuration";
    
    /**
     * @var string The class of the icon in the UI
     */
    static menuIcon = "fa-solid fa-plus";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "n";

    /**
     * Show the new model form when clicked
     */
    async onClick() {
        this.application.modelManager.showNewModel();
    }
}

export { NewModelController as MenuController }
