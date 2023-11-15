/** @module forms/enfugue/theme */
import { FormView } from "../base.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import {
    ColorInputView,
    FontInputView
} from "../input.mjs";

/**
 * Allows a user to configure some aspects of enfugue's appearance
 */
class ThemeFormView extends FormView {
    /**
     * @var object Grouped fields
     */
    static fieldSets = {
        "Theme Colors": {
            "themeColorPrimary": {
                "class": ColorInputView,
                "label": "Primary"
            },
            "themeColorSecondary": {
                "class": ColorInputView,
                "label": "Secondary"
            },
            "themeColorTertiary": {
                "class": ColorInputView,
                "label": "Tertiary"
            }
        },
        "Dark Colors": {
            "darkColor": {
                "class": ColorInputView,
                "label": "Dark"
            },
            "darkerColor": {
                "class": ColorInputView,
                "label": "Darker"
            },
            "darkestColor": {
                "class": ColorInputView,
                "label": "Darkest"
            }
        },
        "Light Colors": {
            "lightColor": {
                "class": ColorInputView,
                "label": "Light"
            },
            "lighterColor": {
                "class": ColorInputView,
                "label": "Lighter"
            },
            "lightestColor": {
                "class": ColorInputView,
                "label": "Lightest"
            }
        },
        "Fonts": {
            "headerFont": {
                "class": FontInputView,
                "label": "Headers"
            },
            "bodyFont": {
                "class": FontInputView,
                "label": "Body"
            },
            "monospaceFont": {
                "class": FontInputView,
                "label": "Monospace"
            }
        }
    };
};

export { ThemeFormView };
