/** @module forms/input/enfugue/civitiai */
import { FormView } from "../../base.mjs";
import { StringInputView } from "../string.mjs";
import { SelectInputView } from "../enumerable.mjs";
import { CheckboxInputView } from "../bool.mjs";

/**
 * The options for sorting in CivitAI
 */
class CivitAISortInputView extends SelectInputView {
    /**
     * @var object the option values and labels (CivitAI uses sentence cased parameters)
     */
    static defaultOptions = [
        "Highest Rated",
        "Most Downloaded",
        "Newest"
    ];

    /**
     * @var string The default value
     */
    static defaultValue = "Highest Rated";
};

/**
 * The options for time period in CivitAI
 */
class CivitAITimePeriodInputView extends SelectInputView {
    /**
     * @var object the option values and labels
     */
    static defaultOptions = {
        "AllTime": "All Time",
        "Year": "This Year",
        "Month": "This Month",
        "Week": "This Week",
        "Day": "Today"
    };

    /**
     * @var string The default value
     */
    static defaultValue = "Year";
};

export {
    CivitAISortInputView,
    CivitAITimePeriodInputView
};
