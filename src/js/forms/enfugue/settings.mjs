/** @module forms/enfugue/settings */
import { FormView } from "../base.mjs";
import { 
    ControlNetModelInputView,
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
        "GPU": {
            "gpu": {
                "label": "Device Index",
                "class": NumberInputView,
                "config": {
                    "value": 0,
                    "min": 0,
                    "step": 1,
                    "tooltip": "When you have more than one AI-capable GPU connected to your computer, you may wish to run ENFUGUE on a GPU other than the primary (first) device. Use this value to change the index of the GPU, starting from 0 for the first."
                }
            }
        },
        "Diffusion": {
            "sequential": {
                "label": "Use Sequential Model Loading",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When checked, the individuals components that make up a diffusion pipeline will be loaded when they are needed and unloaded afterwards. This provides the lowest memory footprint available by Enfugue, but individual images can take longer to generate."
                }
            },
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
                "class": ControlNetModelInputView
            },
            "canny_xl": {
                "label": "Canny Edge XL",
                "class": ControlNetModelInputView
            },
            "hed": {
                "label": "Holistically-Nested Edge Detection (HED)",
                "class": ControlNetModelInputView
            },
            "hed_xl": {
                "label": "Holistically-Nested Edge Detection (HED) XL",
                "class": ControlNetModelInputView
            },
            "pidi": {
                "label": "Soft Edge Detection (PIDI)",
                "class": ControlNetModelInputView
            },
            "pidi_xl": {
                "label": "Soft Edge Detection (PIDI) XL",
                "class": ControlNetModelInputView
            },
            "mlsd": {
                "label": "Mobile Line Segment Detection (MLSD)",
                "class": ControlNetModelInputView
            },
            "mlsd_xl": {
                "label": "Mobile Line Segment Detection (MLSD) XL",
                "class": ControlNetModelInputView
            },
            "line": {
                "label": "Line Art",
                "class": ControlNetModelInputView
            },
            "line_xl": {
                "label": "Line Art XL",
                "class": ControlNetModelInputView
            },
            "anime": {
                "label": "Anime Line Art",
                "class": ControlNetModelInputView
            },
            "anime_xl": {
                "label": "Anime Line Art XL",
                "class": ControlNetModelInputView
            },
            "scribble": {
                "label": "Scribble",
                "class": ControlNetModelInputView
            },
            "scribble_xl": {
                "label": "Scribble XL",
                "class": ControlNetModelInputView
            },
            "depth": {
                "label": "Depth Detection (MiDaS)",
                "class": ControlNetModelInputView
            },
            "depth_xl": {
                "label": "Depth Detection (MiDaS) XL",
                "class": ControlNetModelInputView
            },
            "normal": {
                "label": "Normal Detection (Estimate)",
                "class": ControlNetModelInputView
            },
            "normal_xl": {
                "label": "Normal Detection (Estimate) XL",
                "class": ControlNetModelInputView
            },
            "pose": {
                "label": "Pose Detection (DWPose/OpenPose)",
                "class": ControlNetModelInputView
            },
            "pose_xl": {
                "label": "Pose Detection (DWPose/OpenPose) XL",
                "class": ControlNetModelInputView
            },
            "qr": {
                "label": "QR Code (QR Monster)",
                "class": ControlNetModelInputView
            },
            "qr_xl": {
                "label": "QR Code (QR Monster) XL",
                "class": ControlNetModelInputView,
            },
            "tile": {
                "label": "Tile",
                "class": ControlNetModelInputView
            },
            "tile_xl": {
                "label": "Tile XL",
                "class": ControlNetModelInputView
            },
            "inpaint": {
                "label": "Inpaint",
                "class": ControlNetModelInputView
            },
            "inpaint_xl": {
                "label": "Inpaint XL",
                "class": ControlNetModelInputView
            }
        }
    };

    /**
     * Collapse ControlNets by default
     */
    static collapseFieldSets = ["ControlNets"];
};

export { SystemSettingsFormView };
