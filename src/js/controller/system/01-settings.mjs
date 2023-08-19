/** @module controller/system/01-settings */
import { isEmpty } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { SystemSettingsFormView } from "../../forms/enfugue/settings.mjs";

/**
 * The systems setting controll just opens up the system settings form(s)
 */
class SystemSettingsController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Settings";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-gear";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "t";

    /**
     * @var int The width of the settings window
     */
    static settingsWindowWidth = 460;

    /**
     * @var int The height of the settings window
     */
    static settingsWindowHeight = 600;

    /**
     * Gets the settings from the API
     */
    getSettings() {
        return this.model.get("/settings");
    }

    /**
     * Updates settings through the API
     */
    updateSettings(newSettings) {
        return this.model.post("/settings", null, null, newSettings);
    }

    /**
     * Gets the System Setting view
     */
    async getSettingsView() {
        let currentSettings = await this.getSettings();
        if (isEmpty(this.settingsView)) {
            this.settingsView = new SystemSettingsFormView(this.config, currentSettings);
            this.settingsView.onSubmit(async (values) => {
                if (this.showWarning && !(await this.confirm(
                    "Changing settings will terminate any active invocation. Continue?"
                ))) {
                    this.settingsView.enable();
                    return;
                }
                try {
                    await this.updateSettings(values);
                    if (values.auth !== currentSettings.auth) {
                        this.notify("info", "Settings Updated", "Successfully updated settings. Reloading page to update authentication data…");
                        setTimeout(() => window.location.href = "/logout", 2000);
                    } else {
                        this.notify("info", "Settings Updated", "Successfully updated settings.");
                    }
                } catch(e) {
                    let errorMessage = `${e}`;
                    if (!isEmpty(e.detail)) {
                        errorMessage = e.detail;
                    } else if (!isEmpty(e.title)) {
                        errorMessage = e.title;
                    }
                    this.notify("error", "Couldn't Update Settings", errorMessage);
                }
                this.settingsView.enable();
            });
        } else {
            this.settingsView.setValues(currentSettings);
        }
        return this.settingsView;
    };

    /**
     * Builds the manager if not yet built.
     */
    async showSettingsManager() {
        if (isEmpty(this.settingsManager)) {
            this.settingsManager = await this.spawnWindow(
                "Settings",
                await this.getSettingsView(),
                this.constructor.settingsWindowWidth,
                this.constructor.settingsWindowHeight
            );
            this.settingsManager.onClose(() => { this.settingsManager = null; });
        } else {
            this.settingsManager.focus();
        }
    }

    /**
     * When clicked, show settings window.
     */
    async onClick() {
        this.showSettingsManager();
    }

    /**
     * On initialization, bind listeners.
     */
     async initialize() {
        this.subscribe("invocationBegin", () => { this.showWarning = true; });
        this.subscribe("invocationComplete", () => { this.showWarning = false; });
        this.subscribe("invocationError", () => { this.showWarning = false; });
     }
};

export { SystemSettingsController as MenuController };
