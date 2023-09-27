/** @module forms/enfugue/settings */
import { FormView } from "../base.mjs";
import { 
    CheckboxInputView,
    NumberInputView,
    SelectInputView,
    StringInputView,
    SliderPreciseInputView,
    PipelineSwitchModeInputView,
    PipelineCacheModeInputView,
    PipelinePrecisionModeInputView,
    PipelineInpaintingModeInputView,
} from "../input.mjs";

class ControlNetPathInputView extends StringInputView {
    /**
     * @var string The placeholder
     */
    static placeholder = "File or URL or Repository";

    /**
     * @var string the tooltip
     */
    static tooltip = "Enter the path to the Huggingface Diffusers repository containing the configuration for a ControlNet to use for each supported ControlNet type.<br /><br />You can also directly point to a checkpoint or other pretrained file. See https://huggingface.co for more details.";
}

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
            "inpainting": {
                "label": "Inpainting Mode",
                "class": PipelineInpaintingModeInputView
            },
            "intermediate_steps": {
                "label": "Intermediate Steps",
                "class": SliderPreciseInputView,
                "config": {
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "The number of steps to wait before decoding the image is at is being diffused. Setting this to a lower number will give you more feedback as your image is being generated, but will also increase the time it takes to make images overall. Set this to 0 to disable intermediates entirely, which can aid in making inference faster and reducing memory usage."
                }
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
        },
        "ControlNets": {
            "canny": {
                "label": "Canny Edge",
                "class": ControlNetPathInputView
            },
            "canny_xl": {
                "label": "Canny Edge XL",
                "class": ControlNetPathInputView
            },
            "hed": {
                "label": "Holistically-Nested Edge Detection (HED)",
                "class": ControlNetPathInputView
            },
            "hed_xl": {
                "label": "Holistically-Nested Edge Detection (HED) XL",
                "class": ControlNetPathInputView
            },
            "pidi": {
                "label": "Soft Edge Detection (PIDI)",
                "class": ControlNetPathInputView
            },
            "pidi_xl": {
                "label": "Soft Edge Detection (PIDI) XL",
                "class": ControlNetPathInputView
            },
            "mlsd": {
                "label": "Mobile Line Segment Detection (MLSD)",
                "class": ControlNetPathInputView
            },
            "mlsd_xl": {
                "label": "Mobile Line Segment Detection (MLSD) XL",
                "class": ControlNetPathInputView
            },
            "line": {
                "label": "Line Art",
                "class": ControlNetPathInputView
            },
            "line_xl": {
                "label": "Line Art XL",
                "class": ControlNetPathInputView
            },
            "anime": {
                "label": "Anime Line Art",
                "class": ControlNetPathInputView
            },
            "anime_xl": {
                "label": "Anime Line Art XL",
                "class": ControlNetPathInputView
            },
            "scribble": {
                "label": "Scribble",
                "class": ControlNetPathInputView
            },
            "scribble_xl": {
                "label": "Scribble XL",
                "class": ControlNetPathInputView
            },
            "depth": {
                "label": "Depth Detection (MiDaS)",
                "class": ControlNetPathInputView
            },
            "depth_xl": {
                "label": "Depth Detection (MiDaS) XL",
                "class": ControlNetPathInputView
            },
            "normal": {
                "label": "Normal Detection (Estimate)",
                "class": ControlNetPathInputView
            },
            "normal_xl": {
                "label": "Normal Detection (Estimate) XL",
                "class": ControlNetPathInputView
            },
            "pose": {
                "label": "Pose Detection (DWPose/OpenPose)",
                "class": ControlNetPathInputView
            },
            "pose_xl": {
                "label": "Pose Detection (DWPose/OpenPose) XL",
                "class": ControlNetPathInputView
            },
            "qr": {
                "label": "QR Code (QR Monster)",
                "class": ControlNetPathInputView
            },
            "qr_xl": {
                "label": "QR Code (QR Monster) XL",
                "class": ControlNetPathInputView,
            },
            "tile": {
                "label": "Tile",
                "class": ControlNetPathInputView
            },
            "tile_xl": {
                "label": "Tile XL",
                "class": ControlNetPathInputView
            },
            "inpaint": {
                "label": "Inpaint",
                "class": ControlNetPathInputView
            },
            "inpaint_xl": {
                "label": "Inpaint XL",
                "class": ControlNetPathInputView
            }
        }
    };

    /**
     * Collapse ControlNets by default
     */
    static collapseFieldSets = ["ControlNets"];
};

export { SystemSettingsFormView };
