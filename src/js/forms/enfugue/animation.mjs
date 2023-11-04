/** @module forms/enfugue/animation */
import { FormView } from "../base.mjs";
import {
    NumberInputView,
    CheckboxInputView,
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
            /*
            "animationRate": {
                "label": "Animation Rate",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "step": 1,
                    "value": 8,
                    "tooltip": "The number of frames per second the resulting animation should be played at. For example, an animation of 16 frames saved at 8 frames per second will be two seconds long."
                }
            },
            */
            "animationLoop": {
                "label": "Loop Animation",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "When enabled, the animation will loop seamlessly such that there will be no hitches when the animation is repeated. This increases render time."
                }
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
                    "value": 32,
                    "tooltip": "How long position encoding data should be after truncating and scaling. For example, if you truncate position data to 16 frames and scale position data to 32 frames, you will have removed the final 8 frames of training data, then altered the timescale of the animation by one half - i.e., the animation will appear about twice as slow. This feature is experimental and may result in strange movement."
                }
            },
            /*
            "animationInterpolation": {
                "label": "Frame Interpolation",
                "class": AnimationInterpolationStepsInputView,
            }
            */
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

        if (this.values.animationLoop) {
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
}

export { AnimationFormView };
