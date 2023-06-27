import { MenuController } from "../menu.mjs";
import { promptFiles, isEmpty } from "../../base/helpers.mjs";

class DrawScribbleController extends MenuController {
    static menuName = "Draw Scribble";
    static menuIcon = "fa-solid fa-pencil";

    async onClick() {
        this.images.addScribbleNode();
    }
}

export { DrawScribbleController as ToolbarController };
