/** @module controller/toolbar/01-load-image */
import { MenuController } from "../menu.mjs";
import { truncate, promptFiles, isEmpty } from "../../base/helpers.mjs";

/**
 * The load image controller allows for selecting an image from a file dialog
 */
class LoadImageController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Load Image";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-regular fa-image";

    /**
     * When clicked, read file then add image node
     */
    async onClick() {
        let imageToLoad;
        try {
            imageToLoad = await promptFiles("image/*");
        } catch(e) {
            // No files selected
        }
        if (!isEmpty(imageToLoad)) {
            let reader = new FileReader();
            reader.onload = (e) => this.images.addImageNode(e.target.result, truncate(imageToLoad.name, 16));
            reader.readAsDataURL(imageToLoad);
        }
    }
}

export { LoadImageController as ToolbarController };
