/** @module controller/system/02-users */
import { MenuController } from "../menu.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { ParentView } from "../../view/base.mjs";
import { ModelTableView } from "../../view/table.mjs";
import { 
    CheckboxInputView, 
    StringInputView,
    PasswordInputView,
    ButtonInputView
} from "../../view/forms/input.mjs";

/**
 * A small view extension for the new user button
 */
class NewUserButtonInputView extends ButtonInputView {
    /**
     * @var string custom classname
     */
    static className = "new-user-input-view";

    /**
     * @var string button inputs use their value as label
     */
    static defaultValue = "New User";
}

/**
 * The form gathers all inputs for the user
 */
class UserForm extends FormView {
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

/**
 * Extend the user table to control buttons when needed
 */
class UsersTableView extends ModelTableView {
    /**
     * @var object default columns
     */
    static columns = {
        "username": "Username",
        "name": "Name",
        "type": "Type"
    };

    /**
     * @var object default formatters
     */
    static columnFormatters = {
        "type": (_, datum) => {
            let isAdmin = isEmpty(datum.permission_groups)
                ? false
                : datum.permission_groups.reduce(
                    (carry, group) => carry || (group.group[0].label === "admin"), 
                    false
                );
            return isAdmin ? "Admin" : "User";
        },
        "name": (_, datum) => {
            return `${datum.first_name} ${datum.last_name}`.trim();
        }
    };

    /**
     * @var array fields that can be searched on
     */
    static searchFields = ["username", "first_name", "last_name"];
}

/**
 * The systems setting controll just opens up the system user form(s)
 */
class UsersController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Users";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-users";

    /**
     * @var int The width of the user window
     */
    static userWindowWidth = 600;

    /**
     * @var int The height of the user window
     */
    static userWindowHeight = 500;

    /**
     * Gets the user from the API
     */
    getUser() {
        return this.model.get("/user");
    }

    /**
     * Updates user through the API
     */
    updateUser(newUser) {
        return this.model.post("/user", null, null, newUser);
    }

    /**
     * Override isEnabled to hide when using noauth
     */
    static isDisabled() {
        return super.isDisabled() || isEmpty(window.enfugue) || window.enfugue.user === "noauth";
    }

    /**
     * Gets the System Setting view
     */
    async getUserView() {
        if (isEmpty(this.tableView)) {
            this.tableView = new UsersTableView(this.config, this.model.User);
            this.tableView.addButton("Edit", "fa-solid fa-edit", async (row) => {
                let userData = row.getAttributes();
                userData.admin = isEmpty(row.permission_groups)
                    ? false
                    : row.permission_groups.reduce(
                        (carry, group) => carry || (group.group[0].label === "admin"), 
                        false
                    );
                let userForm = new UserForm(this.config, userData),
                    userWindow = await this.spawnWindow(
                        `Edit ${row.username}`,
                        userForm,
                        this.constructor.userWindowWidth,
                        this.constructor.userWindowHeight
                    );

                userForm.onSubmit(async (updatedValues) => {
                    userForm.clearError();
                    try {
                        if (!isEmpty(updatedValues.new_password) && !isEmpty(updatedValues.repeat_password)) {
                            if (updatedValues.new_password !== updatedValues.repeat_password) {
                                throw "Passwords do not match.";
                            }
                            row.stageChanges({
                                "new_password": updatedValues.new_password,
                                "repeat_password": updatedValues.repeat_password
                            });
                        }
                        row.first_name = updatedValues.first_name;
                        row.last_name = updatedValues.last_name;
                        if (row.username !== "enfugue") {
                            row.stageChange("admin", updatedValues.admin);
                        }
                        await row.save();
                        this.notify("info", "Success", `User ${row.username} updated.`);
                        userWindow.remove();
                        if (!isEmpty(this.tableView)) {
                            this.tableView.requery();
                        }
                    } catch(e) {
                        let errorMessage = `${e}`;
                        if (!isEmpty(e.detail)) errorMessage = e.detail;
                        else if (!isEmpty(e.title)) errorMessage = e.title;
                        userForm.setError(errorMessage);
                        userForm.enable();
                    }
                });
                userForm.onCancel(() => userWindow.remove());
            });
            this.tableView.addButton("Delete", "fa-solid fa-trash", async (row) => {
                if (row.username === "enfugue") {
                    return this.notify("warn", "Forbidden", "Deleting the default user is not allowed.");
                }
                await row.delete();
                if (!isEmpty(this.tableView)) {
                    this.tableView.requery();
                }
            });
        }
        if (isEmpty(this.newUserView)) {
            this.newUserView = new NewUserButtonInputView(this.config);
            this.newUserView.onChange(async () => {
                if (isEmpty(this.newUserWindow)) {
                    let newUserForm = new UserForm(this.config);
                    this.newUserWindow = await this.spawnWindow(
                        "New User",
                        newUserForm,
                        this.constructor.userWindowWidth,
                        this.constructor.userWindowHeight
                    );
                    newUserForm.onSubmit(async (newUser) => {
                        newUserForm.clearError();
                        try {
                            if (isEmpty(newUser.new_password) || isEmpty(newUser.repeat_password)) {
                                throw "Password is required.";
                            }
                            if (newUser.new_password !== newUser.repeat_password) {
                                throw "Passwords do not match.";
                            }
                            await this.model.User.create(newUser);
                            this.notify("info", "Success", `User ${newUser.username} created.`);
                            this.newUserWindow.remove();
                            if (!isEmpty(this.tableView)) {
                                this.tableView.requery();
                            }
                        } catch(e) {
                            let errorMessage = `${e}`;
                            if (!isEmpty(e.detail)) errorMessage = e.detail;
                            else if (!isEmpty(e.title)) errorMessage = e.title;
                            newUserForm.setError(errorMessage);
                            newUserForm.enable();
                        }
                    });
                    newUserForm.onCancel(() => this.newUserWindow.remove());
                    this.newUserWindow.onClose(() => { this.newUserWindow = null; });
                } else {
                    this.newUserWindow.focus();
                }
            });
        }
        let parentView = new ParentView(this.config);
        await parentView.addChild(this.tableView);
        await parentView.addChild(this.newUserView);
        return parentView;
    };

    /**
     * Builds the manager if not yet built.
     */
    async showUserManager() {
        if (isEmpty(this.userManager)) {
            this.userManager = await this.spawnWindow(
                "Users",
                await this.getUserView(),
                this.constructor.userWindowWidth,
                this.constructor.userWindowHeight
            );
            this.userManager.onClose(() => { this.userManager = null; });
        } else {
            this.userManager.focus();
        }
    }

    /**
     * When clicked, show user window.
     */
    async onClick() {
        this.showUserManager();
    }
};

export { UsersController as MenuController };
