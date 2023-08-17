/** @module forms/enfugue/files */
import { FormView } from "../base.mjs";
import { FileInputView, StringInputView } from "../input.mjs";

/**
 * Create a form to input filename
 */
class FileNameFormView extends FormView {
    /**
     * @var object The field sets
     */
    static fieldSets = {
        "File Name": {
            "filename": {
                "class": StringInputView,
                "config": {
                    "required": true,
                    "value": "Enfugue Project"
                }
            }
        }
    };
};

/**
 * The DirectoryFormView allows changing the filesystem location for a directory
 */
class DirectoryFormView extends FormView {
    /**
     * @var bool Enable canceling
     */
    static showCancel = true;

    /**
     * @var object Only a single fieldset
     */
    static fieldSets = {
        "Directory": {
            "directory": {
                "class": StringInputView,
                "config": {
                    "required": true,
                    "placeholder": "C:\\Users\\MyUser\\..."
                }
            }
        }
    };
};

/**
 * The FileFormView allows uploading a single file to a directory
 */
class FileFormView extends FormView {
    /**
     * @var bool Enable canceling
     */
    static showCancel = true;

    /**
     * @var object Only a single fieldset
     */
    static fieldSets = {
        "File": {
            "file": {
                "class": FileInputView,
                "config": {
                    "required": true
                }
            }
        }
    };
};

export {
    DirectoryFormView,
    FileNameFormView,
    FileFormView
};
