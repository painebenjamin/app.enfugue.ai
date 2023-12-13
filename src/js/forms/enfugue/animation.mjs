/** @module forms/enfugue/animation */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    SelectInputView,
    CheckboxInputView,
    ImageFileInputView,
    AnimationLoopInputView,
    AnimationInterpolationStepsInputView,
} from "../input.mjs";

/**
 * The AnimationFormView gathers inputs for AnimateDiff animation
 */
class AnimationFormView extends FormView {
    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;

    /**
     * @var bool Start collapsed
     */
    static collapseFieldSets = true;

    /**
     * @var object All the inputs in this controller
     */
    static fieldSets = {
        "Animation": {
            "animationEnabled": {
                "label": "Enable Animation",
                "class": CheckboxInputView,
            },
            "animationFrames": {
                "label": "Animation Frames",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "step": 1,
                    "value": 16,
                    "tooltip": "The number of animation frames the overall animation should be. Divide this number by the animation rate to determine the overall length of the animation in seconds."
                }
            },
            "animationLoop": {
                "label": "Loop Animation",
                "class": AnimationLoopInputView
            },
            "animationSlicing": {
                "label": "Use Frame Attention Slicing",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "Similar to slicing along the width or height of an image, when using frame slicing, only a portion of the overall animation will be rendered at once. This will reduce the memory required for long animations, but make the process of creating it take longer overall.<br /><br />Since the animation model is trained on short burts of animation, this can help improve the overall coherence and motion of an animation as well."
                }
            },
            "animationSize": {
                "label": "Frame Window Size",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "max": 64,
                    "value": 16,
                    "tooltip": "This is the number of frames to render at once when used sliced animation diffusion. Higher values will require more VRAM, but reduce the overall number of slices needed to render the final animation."
                }
            },
            "animationStride": {
                "label": "Frame Window Stride",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 32,
                    "value": 8,
                    "tooltip": "This is the numbers of frames to move the frame window by when using sliced animation diffusion. It is recommended to keep this value at least two fewer than the animation engine size, as that will leave at least two frames of overlap between slices and ease the transition between them."
                }
            },
            "animationMotionScaleEnabled": {
                "label": "Use Motion Attention Scaling",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, a scale will be applied to the influence of motion data on the animation that is proportional to the ratio between the motion training resolution and the image resolution. This will generally increase the amount of motion in the final animation."
                }
            },
            "animationMotionScale": {
                "label": "Motion Attention Scale Multiplier",
                "class": NumberInputView,
                "config": {
                    "min": 0.0,
                    "step": 0.01,
                    "value": 1.0,
                    "tooltip": "When using motion attention scaling, this multiplier will be applied to the scaling. You can use this to decrease the amount of motion (values less than 1.0) or increase the amount of motion (values greater than 1.0) in the resulting animation."
                }
            },
            "animationPositionEncodingSliceEnabled": {
                "label": "Use Position Encoding Slicing",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, you can control the length of position encoding data, slicing it short and/or scaling it linearly. Slicing can improve consistency by removing unused late-animation embeddings beyond a frame window, and scaling can act as a timescale modifier."
                }
            },
            "animationPositionEncodingTruncateLength": {
                "label": "Position Encoding Truncate Length",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "max": 24,
                    "value": 16,
                    "tooltip": "Where to end position encoding data. Position tensors are generally 24 frames long, so a value of 16 will remove the final 8 frames of data."
                }
            },
            "animationPositionEncodingScaleLength": {
                "label": "Position Encoding Scale Length",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "value": 24,
                    "tooltip": "How long position encoding data should be after truncating and scaling. For example, if you truncate position data to 16 frames and scale position data to 24 frames, you will have removed the final 8 frames of training data, then altered the timescale of the animation by one half - i.e., the animation will appear about 50% slower. This feature is experimental and may result in strange movement."
                }
            },
            "animationRate": {
                "label": "Frame Rate",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "value": 8,
                    "max": 128,
                    "tooltip": "The frame rate of the output video. Note that the animations are saved as individual frames, not as videos - so this can be changed later without needing to re-process the invocation. Also note that the frame rate of the AI model is fixed at 8 frames per second, so any values higher than this will result in sped-up motion. Use this value in combination with frame interpolation to control the smoothness of the output video."
                }
            },
            "animationInterpolation": {
                "label": "Frame Interpolation",
                "class": AnimationInterpolationStepsInputView
            }
        }
    };

    /**
     * On submit, add/remove CSS for hiding/showing
     */
    async submit() {
        await super.submit();

        if (this.values.animationEnabled) {
            this.removeClass("no-animation");
        } else {
            this.addClass("no-animation");
        }

        if (this.values.animationMotionScaleEnabled) {
            this.removeClass("no-animation-scaling");
        } else {
            this.addClass("no-animation-scaling");
        }

        if (this.values.animationPositionEncodingSliceEnabled) {
            this.removeClass("no-position-slicing");
        } else {
            this.addClass("no-position-slicing");
        }

        let useSlicing = this.values.animationSlicing,
            slicingInput = await this.getInputView("animationSlicing");

        if (this.values.animationLoop === "loop") {
            useSlicing = true;
            slicingInput.setValue(true, false);
            slicingInput.disable();
        } else {
            slicingInput.enable();
        }

        if (useSlicing) {
            this.removeClass("no-animation-slicing");
        } else {
            this.addClass("no-animation-slicing");
        }
    }

    /**
     * On first build, add CSS for hiding defaults
     */
    async build() {
        let node = await super.build();
        if (isEmpty(this.values) || this.values.animationEnabled !== true) {
            node.addClass("no-animation");
        }
        if (isEmpty(this.values) || this.values.animationPositionEncodingSliceEnabled !== true) {
            node.addClass("no-position-slicing");
        }
        return node;
    }
}

/**
 * This form allows selecting specific options for SVD.
 */
class StableVideoDiffusionFormView extends FormView {
    /**
     * @var object Options for SVD
     */
    static fieldSets = {
        "Model": {
            "model": {
                "class": SelectInputView,
                "config": {
                    "options": {
                        "svd": "SVD (18 Frames)",
                        "svd_xt": "SVD-XT (25 Frames)"
                    },
                    "value": "svd",
                }
            }
        },
        "Image": {
            "image": {
                "class": ImageFileInputView,
                "config": {
                    "required": true,
                    "tooltip": "The first image of the animation. The recommended resolution is 1024×576, with some ability to create vertical video at 576×1024. Other resolutions may work, but some can produce errors."
                }
            }
        },
        "Generation": {
            "motion_bucket_id": {
                "label": "Motion Bucket ID",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 512,
                    "value": 127,
                    "step": 1,
                    "tooltip": "Approximately represents the amount of motion in the frame, using values from 1 to 255. Higher values are accepted with unpredictable results."
                }
            },
            "seed": {
                "label": "Seed",
                "class": NumberInputView,
                "config": {
                    "step": 1,
                    "tooltip": "The initial seed. Seed this to a specific number for repeatable generations, or leave blank for random."
                }
            }
        },
        "Tweaks": {
            "num_inference_steps": {
                "label": "Inference Steps",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 200,
                    "value": 25,
                    "step": 1,
                    "tooltip": "The number of steps to run through the UNet. Defaults to 25."
                }
            },
            "min_guidance_scale": {
                "label": "Minimum Guidance",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 1.0,
                    "step": 0.01,
                    "tooltip": "The starting guidance scale. This will increase linearly to the ending scale over the course of inference."
                }
            },
            "max_guidance_scale": {
                "label": "Maximum Guidance",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 3.0,
                    "step": 0.01,
                    "tooltip": "The ending guidance scale."
                }
            },
            "fps": {
                "label": "FPS",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 60,
                    "value": 7,
                    "step": 1,
                    "tooltip": "The number of frames per second for inference. Note that this is slightly different from the usual frame rate in that the diffusion model uses this value as a parameter."
                }
            },
            "noise_aug_strength": {
                "label": "Noise Strength",
                "class": NumberInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.02,
                    "step": 0.01,
                    "tooltip": "The factor when adding noise to the initial image. The recommended value is 0.02."
                }
            }
        },
        "Post-Processing": {
            "reflect": {
                "label": "Reflect Animation",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the animation will play in reverse after playing normally. Some interpolated frames will be added at the beginning and end to ease the motion bounce."
                }
            },
            "interpolate_frames": {
                "label": "Frame Interpolation",
                "class": AnimationInterpolationStepsInputView
            },
            "frame_rate": {
                "label": "Frame Rate",
                "class": NumberInputView,
                "config": {
                    "min": 4,
                    "max": 120,
                    "value": 8,
                    "tooltip": "The final frame rate of the output video after interpolation."
                }
            }
        }
    };
};

class QuickStableVideoDiffusionFormView extends StableVideoDiffusionFormView {
    /**
     * @var object Hide the image input
     */
    static fieldSetConditions = {
        "Image": () => false
    }
};

export {
    AnimationFormView,
    StableVideoDiffusionFormView,
    QuickStableVideoDiffusionFormView
};
