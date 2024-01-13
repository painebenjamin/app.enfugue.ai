/** @module controller/help/01-about */
import { ElementBuilder } from "../../base/builder.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { ControlsView } from "../../view/controls.mjs";

const E = new ElementBuilder();

/**
 * The controls controller just shows the controls
 */
class ControlsController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Controls";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-gamepad";

    /**
     * @var int The width of the controls window
     */
    static controlsWindowWidth = 500;

    /**
     * @var int The height of the controls window
     */
    static controlsWindowHeight = 700;
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "a";

    /**
     * On click, reset state
     */
    async onClick() {
        if (isEmpty(this.controlsWindow)) {
            this.controlsWindow = await this.spawnWindow(
                "Controls",
                await (new ControlsView(this.config)).getNode(),
                this.constructor.controlsWindowWidth,
                this.constructor.controlsWindowHeight
            );
            this.controlsWindow.onClose(() => { this.controlsWindow = null; });
        } else {
            this.controlsWindow.focus();
        }
    }
};

export { ControlsController as MenuController };
