/** @module controller/system/01-settings */
import { MenuController } from "../menu.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { CheckboxInputView, NumberInputView } from "../../view/forms/input.mjs";

/**
 * This class assembles all settings manageable from the UI
 */
class SystemSettingsView extends FormView {
    /**
     * @var object The field sets for this form
     */
    static fieldSets = {
        "Safety": {
            "safe": {
                "label": "Use Safety Checker",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "The Safety Checker will evaluate images after they have been diffused to determine if they contain explicit content. If they do, the image will not be saved, and a blank image will be returned."
                }
            }
        },
        "Users": {
            "auth": {
                "label": "Use Authentication",
                "class": CheckboxInputView,
                "config": {
                    "value": false,
                    "tooltip": "When checked, you will be required to use a username and password to login to the application before you will be able to use it."
                }
            }
        },
        "Diffusion": {
            "maxQueuedInvocations": {
                "label": "Queue Size",
                "class": NumberInputView,
                "config": {
                    "required": true,
                    "value": 2,
                    "min": 0,
                    "step": 1,
                    "tooltip": "The maximum number of invocations to queue. Invocations are queued when a user tries to use the diffusion engine while it is currently in use with another invocation. If there are no queue slots available, the user who attempts to use the engine will receive an error."
                }
            }
        },
        "Downloads": {
            "maxConcurrentDownloads": {
                "label": "Concurrent Downloads",
                "class": NumberInputView,
                "config": {
                    "required": true,
                    "value": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "The maximum number of downloads to allow at once."
                }
            },
            "maxQueuedDownloads": {
                "label": "Queue Size",
                "class": NumberInputView,
                "config": {
                    "required": true,
                    "value": 2,
                    "min": 0,
                    "step": 1,
                    "max": 10,
                    "tooltip": "The maximum number of downloads to queue."
                }
            }
        }
    };
};

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
     * @var int The width of the settings window
     */
    static settingsWindowWidth = 600;

    /**
     * @var int The height of the settings window
     */
    static settingsWindowHeight = 510;

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
        let currentSettings = await this.getSettings(),
            settingsValues = {
                "safe": currentSettings.safe,
                "auth": currentSettings.auth,
                "maxQueuedInvocations": currentSettings.max_queued_invocations,
                "maxQueuedDownloads": currentSettings.max_queued_downloads,
                "maxConcurrentDownloads": currentSettings.max_concurrent_downloads
            };
        if (isEmpty(this.settingsView)) {
            this.settingsView = new SystemSettingsView(this.config, settingsValues);
            this.settingsView.onSubmit(async (values) => {
                try {
                    await this.updateSettings({
                        "safe": values.safe,
                        "auth": values.auth,
                        "max_queued_invocations": values.maxQueuedInvocations,
                        "max_queued_downloads": values.maxQueuedDownloads,
                        "max_concurrent_downloads": values.maxConcurrentDownloads
                    });
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
            this.settingsView.setValues(settingsValues);
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
};

export { SystemSettingsController as MenuController };
