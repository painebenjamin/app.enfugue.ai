/** @module forms/enfugue/civitiai */
import { FormView } from "../base.mjs";
import {
    CivitAISortInputView,
    CivitAITimePeriodInputView,
    CheckboxInputView,
    StringInputView
} from "../input.mjs";

/**
 * This form gathers together all the filter options
 */
class CivitAISearchOptionsFormView extends FormView {
    /**
     * @var object The fieldset labels and config
     */
    static fieldSets = {
        "Sorting": {
            "sort": {
                "label": "Sort Method",
                "class": CivitAISortInputView
            },
            "period": {
                "label": "Time Period",
                "class": CivitAITimePeriodInputView
            }
        },
        "Filters": {
            "commercial": {
                "label": "Only Show Models Allowing Commercial Use",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Checking this box will ensure all results have at least the 'Image' commercial use status from CivitAI. Some models may additionally authorize you to distribute or modify the models, but not all - review the details on each result before making such a determination."
                }
            },
            "nsfw": {
                "label": "Show NSFW Models and Images",
                "class": CheckboxInputView,
                "config": {
                    "tooltip": "Checking this box will show <strong>NSFW (Not Safe For Work)</strong> content of a sexual or explicit nature.<br /><br /><em>Note:</em> If safety checking is enabled on a system-wide level, NSFW results will never be returned, regardless of whether or not this box is checked."
                }
            },
            "search": {
                "class": StringInputView,
                "config": {
                    "placeholder": "Search by name, user, description, etc."
                }
            }
        }
    };
};

export { CivitAISearchOptionsFormView };
