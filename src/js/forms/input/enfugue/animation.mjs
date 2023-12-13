/** @module forms/input/enfugue/animation */
import { SelectInputView } from "../enumerable.mjs";
import { RepeatableInputView } from "../parent.mjs";
import { NumberInputView } from "../numeric.mjs";

/**
 * This class allows selecting looping options (or none)
 */
class AnimationLoopInputView extends SelectInputView {
    /**
     * @var bool enable selecting null
     */
    static allowEmpty = true;

    /**
     * @var string Text to show in null option
     */
    static placeholder = "No Looping";

    /**
     * @var object Selectable options
     */
    static defaultOptions = {
        "reflect": "Reflect",
        "loop": "Loop"
    };

    /**
     * @var string tooltip to display
     */
    static tooltip = "When enabled the animation will loop seamlessly such that there will be no hitches when the animation is repeated. <strong>Reflect</strong> will add a reverse version of the animation at the end, with interpolation to ease the inflection points. <strong>Loop</strong> will create a properly looking animation through sliced diffusion; this will increase render time.";
}

/**
 * Provides a repeatable input for interpolation steps
 */
class AnimationInterpolationStepsInputView extends NumberInputView {
    /**
     * @var int Minimum value, 0 is disabled
     */
    static min = 0;

    /**
     * @var string tooltip
     */
    static tooltip = "Interpolation is the process of trying to calculate a frame between two other frames such that when the calculated frame is placed between the two other frames, there appears to be a smooth motion between the three.<br />Specify any number greater than zero to add that many frames in-between every frame, interpolated using FILM - Frame Interpolation for Large Motion, an AI model from Google.";
}

export {
    AnimationLoopInputView,
    AnimationInterpolationStepsInputView,
}
