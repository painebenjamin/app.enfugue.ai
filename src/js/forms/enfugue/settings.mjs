/** @module forms/enfugue/settings */
import { FormView } from "../base.mjs";
import { 
    CheckboxInputView,
    NumberInputView,
    SelectInputView,
    PipelineSwitchModeInputView,
    PipelineCacheModeInputView,
    PipelinePrecisionModeInputView
} from "../input.mjs";

/**
 * This class assembles all settings manageable from the UI
 */
class SystemSettingsFormView extends FormView {
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

export { SystemSettingsFormView };
