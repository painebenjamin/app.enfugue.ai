/** @module forms/enfugue/user */
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../base.mjs";
import { 
    CheckboxInputView, 
    StringInputView,
    PasswordInputView,
    ButtonInputView
} from "../input.mjs";

/**
 * The form gathers all inputs for the user
 */
class UserFormView extends FormView {
    /**
     * @var object The fields
     */
    static fieldSets = {
        "User": {
            "username": {
                "label": "Username",
                "class": StringInputView,
                "config": {
                    "required": true,
                    "editable": false,
                    "pattern": "^[a-zA-Z0-9_]{2,}$",
                    "minlength": 2,
                    "maxlength": 128
                }
            },
            "first_name": {
                "label": "First Name",
                "class": StringInputView
            },
            "last_name": {
                "label": "Last Name",
                "class": StringInputView
            }
        },
        "Permissions": {
            "admin": {
                "label": "Administrator",
                "class": CheckboxInputView
            }
        },
        "Password": {
            "new_password": {
                "label": "New Password",
                "class": PasswordInputView
            },
            "repeat_password": {
                "label": "Repeat Password",
                "class": PasswordInputView
            }
        }
    };

    /**
     * On build, disable permission change if user is default.
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.values)) {
            if (this.values.username === "enfugue") {
                this.inputViews.filter((view) => view.fieldName === "admin")[0].disable();
            }
        }
        return node;
    }
};

export { UserFormView };
