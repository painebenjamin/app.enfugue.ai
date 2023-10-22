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
            "animationChunking": {
                "label": "Use Frame Chunking",
                "class": CheckboxInputView,
                "config": {
                    "value": true,
                    "tooltip": "Similar to chunking along the width or height of an image, when using frame chunking, only a portion of the overall animation will be rendered at once. This will reduce the memory required for long animations, but make the process of creating it take longer overall.<br /><br />Since the animation model is trained on short burts of animation, this can help improve the overall coherence and motion of an animation as well."
                }
            },
            "animationSize": {
                "label": "Animation Engine Size",
                "class": NumberInputView,
                "config": {
                    "min": 8,
                    "max": 64,
                    "value": 16,
                    "tooltip": "This is the number of frames to render at once when used chunked animation diffusion. Higher values will require more VRAM, but reduce the overall number of chunks needed to render the final animation."
                }
            },
            "animationStride": {
                "label": "Animation Engine Stride",
                "class": NumberInputView,
                "config": {
                    "min": 1,
                    "max": 32,
                    "value": 8,
                    "tooltip": "This is the numbers of frames to move the frame window by when using chunked animation diffusion. It is recommended to keep this value at least two fewer than the animation engine size, as that will leave at least two frames of overlap between chunks and ease the transition between them."
                }
            },
            "animationLoop": {
                "label": "Loop Animation",
                "class": AnimationLoopInputView
            },
            "animationInterpolation": {
                "label": "Frame Interpolation",
                "class": AnimationInterpolationStepsInputView,
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

        let useChunking = this.values.animationChunking,
            chunkingInput = await this.getInputView("animationChunking");

        if (this.values.animationLoop === "loop") {
            useChunking = true;
            chunkingInput.setValue(true, false);
            chunkingInput.disable();
        } else {
            chunkingInput.enable();
        }

        if (useChunking) {
            this.removeClass("no-animation-chunking");
        } else {
            this.addClass("no-animation-chunking");
        }
    }
}

export { AnimationFormView };
