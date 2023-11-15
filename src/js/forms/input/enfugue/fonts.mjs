/** @module forms/input/enfugue/fonts */
import { SelectInputView } from "../enumerable.mjs";

/**
 * Allows a user to select between some google fonts
 */
class FontInputView extends SelectInputView {
    /**
     * @var array All font options
     */
    static defaultOptions = [
        "Inconsolata",
        "Lato",
        "Lora",
        "Merriweather",
        "Montserrat",
        "Noto Sans",
        "Open Sans",
        "Poppins",
        "Quicksand",
        "Raleway",
        "Roboto",
        "Ubuntu Mono",
        "Work Sans"
    ];
};

export { FontInputView };
