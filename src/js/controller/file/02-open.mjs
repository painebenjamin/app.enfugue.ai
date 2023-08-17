/** @module controller/file/02-open.mjs */
import { MenuController } from "../menu.mjs";
import { promptFiles, isEmpty } from "../../base/helpers.mjs";

/**
 * The open file controller reads a file and sets state
 */
class OpenFileController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Open";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-file-arrow-up";

    /**
     * When clicked, read file then set state
     */
    async onClick() {
        let fileToRead;
        try {
            fileToRead = await promptFiles("*/json");
        } catch(e) {
            // No files selected
        }
        if (!isEmpty(fileToRead)) {
            let reader = new FileReader();
            reader.onload = (e) => {
                try {
                    let result = JSON.parse(e.target.result);
                    this.application.setState(result);
                } catch(e) {
                    this.notify("error", "Couldn't Load", `${e}`);
                }
            };
            reader.readAsText(fileToRead);
        }
    }
};

export { OpenFileController as MenuController };
