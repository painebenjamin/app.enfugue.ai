/** @module controller/system/01-settings */
import { MenuController } from "../menu.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import {
    CheckboxInputView,
    NumberInputView,
    SelectInputView
} from "../../view/forms/input.mjs";

/**
 * Controls how pipelines are switched
 */
class PipelineSwitchModeInputView extends SelectInputView {
    /**
     * @var object The options for switching
     */
    static defaultOptions = {
        "offload": "Offload to CPU",
        "unload": "Unload"
    };
    
    /**
     * @var string Default to offloading to CPU
     */
    static defaultValue = "offload";

    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Keep in Memory";

    /**
     * @var string The tooltip to show the user
     */
    static tooltip = "When making an image, you may switch between a text or infererence pipeline, an inpainting pipeline, and a refining pipeline. In order to balance time spent moving between disk, system memory (RAM), and graphics memory (VRAM), as well as the amount of those resources consumed, the default setting sends pipelines from VRAM to RAM when not needed, then reloads from RAM when needed.<br/><br/>If you set this to <strong>Unload</strong>, pipelines will be freed from VRAM and not sent to RAM when not needed. This will minimize memory consumption, but increase the time spent loading from disk.<br/><br/>If you set this to <strong>Keep in Memory</strong>, pipelines will never be freed from VRAM, and up to three pipelines will be kept available, elimining swapping time. This should only be used with powerful GPUs with <em>at least</em> 12GB of VRAM for SD 1.5 models, or 24GB of VRAM for SDXL models.";
};

/**
 * Controls how pipelines are cached
 */
class PipelineCacheModeInputView extends SelectInputView {
    /**
     * @var object The options for caching
     */
    static defaultOptions = {
        "xl": "Cache XL Pipelines and TensorRT Pipelines",
        "always": "Cache All Pipelines"
    };
    
    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Cache TensorRT Pipelines";

    /**
     * @var string The text to show to the user
     */
    static tooltip = "Models are distributed as <em>.ckpt</em> or <em>.safetensors</em> files for convenience, but opening them and reading them into memory can take some time. For this reason, a cache can be created that speeds up loading time, but takes up approximately as much space as the original checkpoint did, effectively doubling the space a model would take up.<br/><br/>For technical reasons, TensorRT pipelines must always be cached, so this setting cannot be disabled.<br/><br/>The default setting additionally caches when using SDXL models, as this speeds up loading by several factors.<br/><br/>Change this setting to <strong>Cache All Pipelines</strong> in order to cache every pipeline you load. This takes up the most space, but makes switching pipelines the fastest.";
};

/**
 * Controls how data types are changed
 */
class PipelinePrecisionModeInputView extends SelectInputView {
    /**
     * @var object The options for data types
     */
    static defaultOptions = {
        "full": "Always Use Full Precision"
    };
    
    /**
     * @var bool Allow an empty (null) value
     */
    static allowEmpty = true;

    /**
     * @var string The text to show for the null value
     */
    static placeholder = "Use Half-Precision When Available";
    
    /**
     * @var string The text to show to the user
     */
    static tooltip = "When performing calculations on your GPU, we use floating-point numbers of a certain precision. In some cases we must use full-precision in order to calculate correctly, but in some places this is not necessary and calculations can be performed at half-precision instead, without losing quality. In general, you should only change this setting to 'Always Use Full Precision' when you experience errors during diffusion.";
};

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
            "switch_mode": {
                "label": "Pipeline Switch Mode",
                "class": PipelineSwitchModeInputView
            },
            "cache_mode": {
                "label": "Pipeline Cache Mode",
                "class": PipelineCacheModeInputView
            },
            "precision": {
                "label": "Precision Mode",
                "class": PipelinePrecisionModeInputView
            },
            "max_queued_invocations": {
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
            "max_concurrent_downloads": {
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
            "max_queued_downloads": {
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
    static settingsWindowHeight = 720;

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
            this.settingsView = new SystemSettingsView(this.config, currentSettings);
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
