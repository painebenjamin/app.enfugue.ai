/** @module controller/menu.mjs */
import { isEmpty } from "../base/helpers.mjs";
import { Controller } from "./base.mjs";

/**
 * The menu controller is just a small extension of the controller specifically
 * for registering in a top menu.
 */
class MenuController extends Controller {
    /**
     * @var string The text to display in the menu
     */
    static menuName;

    /**
     * @var string The icon to display in the menu
     */
    static menuIcon;
    
    /**
     * @var string The keyboard shortcut to this menu item. TODO
     */
    static menuShortcut;

    /**
     * Returns true if this controller should not be registered.
     */
    static isDisabled() {
        return isEmpty(this.menuName);
    }
    
    /**
     * The onClick handler should be overridden by implementing controllers.
     */
    async onClick() {
        this.notify("warn", 
            "Unimplemented Menu Controller", 
            "Override the onClick() function to give this menu controller an implementation.");
    }
}

export { MenuController };
