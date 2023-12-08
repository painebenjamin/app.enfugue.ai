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
class AnimationInterpolationStepsInputView extends RepeatableInputView {
    /**
     * @var class member class
     */
    static memberClass = NumberInputView;

    /**
     * @var object config for member class
     */
    static memberConfig = {
        "min": 2,
        "max": 10,
        "step": 1,
        "value": 2
    };

    /**
     * @var string tooltip
     */
    static tooltip = "Interpolation is the process of trying to calculate a frame between two other frames such that when the calculated frame is placed between the two other frames, there appears to be a smooth motion between the three.<br />You can add multiple <strong>interpolation factors</strong>, where a value of <em>2</em> means that there will be two frames for every one frame (thus one frame placed in-between every frame,) a value of <em>3</em> means there will be three frames for every one frame (and thus two frames placed in-between every frame, attempting to maintain a smooth motion across all of them.) Multiple factors will be executed recursively. The smoothest results can be obtained via repeated factors of 2.";

    /**
     * @var string Text to show when no items
     */
    static noItemsLabel = "No Interpolation";

    /**
     * @var string Text to show in the 'add item' button
     */
    static addItemLabel = "Add Interpolation Step";
}

export {
    AnimationLoopInputView,
    AnimationInterpolationStepsInputView,
}
