/** @module forms/enfugue/animation */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    SelectInputView,
    CheckboxInputView,
    ImageFileInputView,
    AnimationLoopInputView,
    AnimationInterpolationStepsInputView,
    AnimationEngineInputView
} from "../input.mjs";

const E = new ElementBuilder();

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
            "animationEngine": {
                "label": "Animation Engine",
                "class": AnimationEngineInputView,
                "config": {
                    "value": "ad_hsxl"
                }
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
            "animationRate": {
                "label": "Frame Rate",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "value": 8,
                    "max": 128,
                    "step": 1,
                    "tooltip": "The frame rate of the output video. Note that the animations are saved as individual frames, not as videos - so this can be changed later without needing to re-process the invocation. Also note that the frame rate of the AI model is fixed at 8 frames per second, so any values higher than this will result in sped-up motion. Use this value in combination with frame interpolation to control the smoothness of the output video."
                }
            },
            "animationDecodeChunkSize": {
                "label": "Frame Decode Chunk",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 512,
                    "value": 1,
                    "step": 1,
                    "tooltip": "The number of frames to decode at once when rendering the final output video. Increasing this number increases VRAM requirements while generally decreasing render time."
                }
            },
            "animationInterpolation": {
                "label": "Frame Interpolation",
                "class": AnimationInterpolationStepsInputView
            },
        },
        "AnimateDiff and HotshotXL": {
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
                    "max": 512,
                    "value": 16,
                    "tooltip": "This is the number of frames to render at once when used sliced animation diffusion. Higher values will require more VRAM, but reduce the overall number of slices needed to render the final animation."
                }
            },
            "animationStride": {
                "label": "Frame Window Stride",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 256,
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
                    "max": 512,
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
            "animationDenoisingIterations": {
                "label": "Denoising Iterations",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 32,
                    "value": 1,
                    "step": 1,
                    "tooltip": "The number of times to perform denoising. If this number is greater than one, a process called ablation occurs, whereby the animation is re-noised and then calculated again using position data from the first generation. This can greatly improve consistency of the final animation at a large cost to inference time.<br/><br/><strong>Note:</strong> not all schedulers are supported when this is enabled. DDIM is recommended."
                }
            }
        },
        "Stable Video Diffusion": {
            "stableVideoReflect": {
                "label": "Reflect Animation",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the animation will play in reverse after playing normally. Some interpolated frames will be added at the beginning and end to ease the motion bounce."
                }
            },
            "stableVideoUseDrag": {
                "label": "Use DragNUWA",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, uses DragNUWA 1.5 to provide controls for directing motion of objects from an image. An interface will be overlaid over the canvas that allows you to draw motion splines."
                }
            },
            "stableVideoModel": {
                "class": SelectInputView,
                "label": "Model",
                "config": {
                    "options": {
                        "svd": "SVD (14 Frames)",
                        "svd_xt": "SVD-XT (21 Frames)"
                    },
                    "value": "svd",
                }
            },
            "stableVideoMotionBucketId": {
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
            "stableVideoFps": {
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
            "stableVideoNoiseAugStrength": {
                "label": "Noise Strength",
                "class": NumberInputView,
                "config": {
                    "min": 0.0,
                    "max": 1.0,
                    "value": 0.02,
                    "step": 0.01,
                    "tooltip": "The factor when adding noise to the initial image. The recommended value is 0.02."
                }
            },
            "stableVideoMinGuidanceScale": {
                "label": "Minimum Guidance",
                "class": NumberInputView,
                "config": {
                    "min": 0,
                    "max": 100,
                    "value": 1.0,
                    "step": 0.01,
                    "tooltip": "The starting guidance scale. This will increase linearly to the ending scale over the course of inference. This overrides the global guidance scale."
                }
            },
            "stableVideoMaxGuidanceScale": {
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
            "stableVideoGaussianSigma": {
                "label": "Gaussian Sigma",
                "class": NumberInputView,
                "config": {
                    "min": 5,
                    "max": 100,
                    "value": 20,
                    "step": 1,
                    "tooltip": "This controls the size of the motion vectors as they move across the image using DragNUWA. The higher this value, the strong the pull each vector will have to pixels around them."
                }
            },
            "stableVideoMotionVectorRepeatMode": {
                "label": "Motion Vector Repeat Mode",
                "class": SelectInputView,
                "config": {
                    "options": {
                        "extend": "Extend",
                        "repeat": "Repeat In-Place",
                        "stretch": "Stretch and Slice",
                    },
                    "value": "extend",
                    "tooltip": "How to handle motion vectors when your animation is longer than 14 frames.<br /><br /><strong>Extend</strong>: For each subsequent 14-frame iteration, the motion vectors will repeat starting from the end of the previous section.<br /><strong>Repeat In-Place</strong>: This is similar to the previous, except the vectors will not be moved to the end of the previous, and instead will repeat from the beginning again.<br /><strong>Stretch and Slice</strong>: This will treat each motion vector as if it goes over the entire animation duration. If you have a 28-frame animation, the first half of the vector would be used for the first 14 frames, and the second half for the latter.<br />"
                }
            }
        }
    };

    /**
     * @var object display conditions for fieldsets
     */
    static fieldSetConditions = {
        "AnimateDiff and HotshotXL": (values) => values.animationEnabled && values.animationEngine === "ad_hsxl",
        "Stable Video Diffusion": (values) => values.animationEnabled && values.animationEngine === "svd"
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
            stableVideoModel = await this.getInputView("stableVideoModel"),
            slicingInput = await this.getInputView("animationSlicing");

        if (this.values.animationLoop === "loop") {
            useSlicing = true;
            slicingInput.setValue(true, false);
            slicingInput.disable();
        } else {
            slicingInput.enable();
            this.addClass("no-animation-slicing");
        }
            
        if (useSlicing) {
            this.removeClass("no-animation-slicing");
        } else {
            this.addClass("no-animation-slicing");
        }

        if (this.values.stableVideoUseDrag) {
            stableVideoModel.setValue("svd", false);
            stableVideoModel.disable();
            this.removeClass("no-motion-vectors");
        } else {
            stableVideoModel.enable();
            this.addClass("no-motion-vectors");
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
        if (isEmpty(this.values) || this.values.animationEngine !== "svd" || !this.values.stableVideoUseDrag) {
            node.addClass("no-motion-vectors");
        }
        return node;
    }
}

export {
    AnimationFormView,
};
