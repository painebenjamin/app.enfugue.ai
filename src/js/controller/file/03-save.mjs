/** @module controller/file/03-save */
import { MenuController } from "../menu.mjs";
import { createEvent, isEmpty } from "../../base/helpers.mjs";
import { FileNameFormView } from "../../forms/enfugue/files.mjs";

/**
 * The save controller allows writing the entire contents of the canvas
 * and options forms to a .json file.
 */
class SaveFileController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Save As";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-file-arrow-down";
    
    /**
     * On click, get state and save
     */
    async onClick() {
        this.application.saveAs(
            "Save Project", 
            JSON.stringify(this.application.getState()),
            "application/json",
            ".json"
        );
    }
}

export { SaveFileController as MenuController };
