/** @module controller/common/theme */
import { isEmpty, kebabCase } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { ThemeFormView } from "../../forms/enfugue/theme.mjs";

const E = new ElementBuilder();

/**
 * This class manages selecting between included and custom themes
 */
class ThemeController extends Controller {
    /**
     * @var int Width of the edit theme window
     */
    static themeWindowWidth = 750;

    /**
     * @var int Height of the edit theme window
     */
    static themeWindowHeight = 450;

    /**
     * Gets the current custom theme
     */
    get customTheme() {
        let storedTheme = this.application.session.getItem("customTheme");
        if (isEmpty(storedTheme)) {
            storedTheme = {};
        }
        return {...this.defaultTheme, ...storedTheme};
    }

    /**
     * Gets the default theme
     */
    get defaultTheme() {
        return this.config.themes[this.config.theme];
    }

    /**
     * Sets the current custom theme
     */
    set customTheme(newTheme) {
        this.application.session.setItem("customTheme", newTheme);
    }

    /**
     * Gets the current theme name
     */
    get theme() {
        let storedTheme = this.application.session.getItem("theme");
        if (isEmpty(storedTheme)) storedTheme = this.config.theme;
        return storedTheme;
    }

    /**
     * Sets the current theme name
     */
    set theme(newTheme) {
        this.application.session.setItem("theme", newTheme);
        this.setTheme(this.getTheme(newTheme));
    }

    /**
     * Sets the theme variables
     */
    setTheme(themeConfig) {
        let element = window.document.documentElement;
        for (let themeVariable in themeConfig) {
            element.style.setProperty(`--${kebabCase(themeVariable)}`, themeConfig[themeVariable]);
        }
    }

    /**
     * Gets theme config by name
     */
    getTheme(themeName) {
        if (themeName === "Custom") {
            return this.customTheme;
        } else {
            return this.config.themes[themeName];
        }
    }

    /**
     * Sets the appropriate theme variables based on current stored/default
     */
    resetTheme() {
        this.setTheme(this.getTheme(this.theme));
    }

    /**
     * Creates the theme editor window
     */
    createThemeEditor() {
        let themeForm = new ThemeFormView(this.config, this.customTheme);
        themeForm.onChange(() => {
            this.setTheme(themeForm.values);
        });
        themeForm.onSubmit(() => {
            this.customTheme = themeForm.values;
            this.publish("customThemeChanged");
        });
        return themeForm;
    }

    /**
     * Shows the edit theme window
     */
    async showEditTheme() {
        if (!isEmpty(this.themeWindow)) {
            this.themeWindow.focus();
            return;
        }
        let themeForm = await this.createThemeEditor();
        this.themeWindow = await this.spawnWindow(
            "Custom Theme",
            themeForm,
            this.constructor.themeWindowWidth,
            this.constructor.themeWindowHeight
        );
        themeForm.onSubmit(() => {
            this.themeWindow.remove();
        });
        this.themeWindow.onClose(() => { 
            delete this.themeWindow;
            this.resetTheme();
        });
    }

    /**
     * On initialization, append menu item
     */
    async initialize() {
        let themeMenu = this.application.menu.getCategory("Theme"),
            addTheme = async (themeName, themeConfig) => {
                let themeItem = await themeMenu.addItem(themeName, null, themeName.substring(0, 1).toLowerCase()),
                    themeNode = await themeItem.getNode(),
                    themeColorPreviews = E.div().class("color-previews").content(
                        E.span().css("background-color", themeConfig.themeColorPrimary),
                        E.span().css("background-color", themeConfig.themeColorSecondary),
                        E.span().css("background-color", themeConfig.themeColorTertiary),
                    );
                themeNode.prepend(themeColorPreviews);
                themeItem.onClick(() => {
                    this.theme = themeName;
                });
                return themeItem;
            };

        for (let themeName in this.config.themes) {
            await addTheme(themeName, this.config.themes[themeName]);
        }

        let customTheme = await addTheme("Custom", this.customTheme);
        customTheme.onClick(() => {
            this.showEditTheme();
        });
        this.subscribe("customThemeChanged", async () => {
            let currentTheme = this.customTheme,
                themePreviewContainer = (await customTheme.getNode()).find(".color-previews");
            themePreviewContainer.getChild(0).css("background-color", currentTheme.themeColorPrimary);
            themePreviewContainer.getChild(1).css("background-color", currentTheme.themeColorSecondary);
            themePreviewContainer.getChild(2).css("background-color", currentTheme.themeColorTertiary);
        });

        this.resetTheme();
    }
};

export { ThemeController };
