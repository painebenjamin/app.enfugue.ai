/** @module view/forms/input/parent */
import { InputView } from "./base.mjs";
import { FormView } from "../base.mjs";
import { ElementBuilder } from "../../../base/builder.mjs";
import { isEmpty } from "../../../base/helpers.mjs";

const E = new ElementBuilder({
    "repeatableItem": "enfugue-repeatable-input-item"
});

/**
 * This parent input item allows for repeating any child input item
 */
class RepeatableInputView extends InputView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-repeatable-input";

    /**
     * @var string The label for the add item button
     */
    static addItemLabel = "+";

    /**
     * @var string The label for the remove item button
     */
    static removeItemLabel = "×";

    /**
     * @var class The class of the member items
     */
    static memberClass = InputView;

    /**
     * @var object Configuration to pass to members when constructed
     */
    static memberConfig = {};

    /**
     * @var int The minimum number of items to be considered acceptable, default none
     */
    static minimumItems = 0;

    /**
     * @var int The maximum number of items, default unbounded
     */
    static maximumItems = Infinity;

    /**
     * Allows for passing member config in constructor
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.inputViews = [];
        let memberConfig = fieldConfig.members || {};
        this.memberConfig = {
            ...this.constructor.memberConfig,
            ...memberConfig
        };
        for (let i = 0; i < this.constructor.minimumItems; i++) {
            this.addInput();
        }
    }

    /**
     * When getting the value, simply return the array of values
     * from each of the repeated child inputs.
     */
    getValue() {
        return this.inputViews.map((view) => view.getValue());
    }

    /**
     * When setting the parent value, iterate through child views
     * and set their values using their setters
     */
    setValue(newValue, triggerChange) {
        let newValueArray = Array.isArray(newValue) ? newValue : [newValue],
            currentValueLength = this.inputViews.length,
            newValueLength = newValueArray.length;
 
        for (let i = 0; i < newValueLength; i++) {
            let inputView;
            if (i >= currentValueLength) {
                inputView = this.addInput();
            } else {
                inputView = this.inputViews[i];
            }
            inputView.setValue(newValueArray[i]);
        }

        if (this.node !== undefined) {
            for (let i = newValueLength; i < currentValueLength; i++) {
                this.node.remove(this.node.getChild(i));
            }
            if (this.inputViews.length <= this.constructor.minimumItems) {
                this.disableRemove();
            } else {
                this.enableRemove();
            }
        }
        super.setValue(newValue, triggerChange);
    }

    /**
     * When disabling, disable child views and add/remove
     */
    disable() {
        super.disable();
        for (let inputView of this.inputViews) {
            inputView.disable();
        }
        if (this.node !== undefined) {
            this.node
                .find("input.add-item")
                .disabled(true)
                .addClass("disabled");
            this.disableRemove();
        }
    }

    /**
     * This function disables removing items only
     */
    disableRemove() {
        if (this.node !== undefined) {
            for (let removeItem of this.node.findAll("input.remove-item")) {
                removeItem.disabled(true).addClass("disabled");
            }
        }
    }

    /**
     * Enables adding, editing and removing items
     */
    enable() {
        super.enable();
        for (let inputView of this.inputViews) {
            inputView.enable();
        }
        if (this.node !== undefined) {
            this.node
                .find("input.add-item")
                .disabled(false)
                .removeClass("disabled");
            this.enableRemove();
        }
    }

    /**
     * Enables removing items only
     */
    enableRemove() {
        if (this.node !== undefined) {
            for (let removeItem of this.node.findAll("input.remove-item")) {
                removeItem.disabled(false).removeClass("disabled");
            }
        }
    }

    /**
     * This is the function called when the add button is clicked.
     */
    addInput(inputConfig) {
        inputConfig = inputConfig || {};
        let newInputView = new this.constructor.memberClass(
            this.config,
            this.inputViews.length,
            { ...this.memberConfig, ...inputConfig }
        );
        newInputView.inputParent = this;
        newInputView.onChange(() => this.changed());
        this.inputViews.push(newInputView);
        if (this.node !== undefined) {
            newInputView.getNode().then((newInputNode) => {
                let addItem = this.node.find("input.add-item"),
                    inputContainer = E.repeatableItem().content(newInputNode),
                    removeInput = E.input()
                        .type("button")
                        .class("remove-item")
                        .value(this.constructor.removeItemLabel);

                removeInput.on("click", async (e) => {
                    this.node.remove(inputContainer);
                    this.inputViews = this.inputViews.filter(
                        (view) => newInputView.fieldName !== view.fieldName
                    );
                    for (let i = 0; i < this.inputViews.length; i++) {
                        this.inputViews[i].fieldName = i;
                    }
                    if (
                        this.inputViews.length <= this.constructor.minimumItems
                    ) {
                        this.disableRemove();
                    }
                    addItem.removeClass("disabled");
                    this.changed();
                });

                if (this.inputViews.length >= this.constructor.maximumItems) {
                    addItem.addClass("disabled");
                }
                if (this.inputViews.length <= this.constructor.minimumItems) {
                    removeInput.disabled(true).addClass("disabled");
                } else {
                    this.enableRemove();
                }
                this.node.insertBefore(
                    addItem,
                    inputContainer.append(removeInput)
                );
            });
        }
        return newInputView;
    }

    /**
     * On build, show any input views added before build and 
     * also show the add item buttons
     */
    async build() {
        let node = await super.build(),
            addItem = E.input()
                .type("button")
                .class("add-item")
                .value(this.constructor.addItemLabel);

        for (let inputView of this.inputViews) {
            let inputContainer = E.repeatableItem().content(
                    await inputView.getNode()
                ),
                removeInput = E.input()
                    .type("button")
                    .class("remove-item")
                    .value(this.constructor.removeItemLabel);

            removeInput.on("click", async (e) => {
                node.remove(inputContainer);
                this.inputViews = this.inputViews.filter(
                    (view) => inputView.fieldName !== view.fieldName
                );
                for (let i = 0; i < this.inputViews.length; i++) {
                    this.inputViews[i].fieldName = i;
                }
                if (this.inputViews.length <= this.constructor.minimumItems) {
                    this.disableRemove();
                }
                addItem.removeClass("disabled");
                this.changed();
            });

            if (this.inputViews.length <= this.constructor.minimumItems) {
                removeInput.disabled(true).addClass("disabled");
            }
            node.append(inputContainer.append(removeInput));
        }

        addItem.on("click", () =>
            addItem.hasClass("disabled") ? 0 : this.addInput() && this.changed()
        );
        node.append(addItem);
        return node;
    }
}

/**
 * This parent input class allows you to declare a form as an input, effectively allowing
 * for multi-part or multi-key inputs.
 */
class FormInputView extends InputView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-form-input-view";
    
    /**
     * @var class the class of the form to instantiate
     */
    static formClass = FormView;

    /**
     * When constructing the input, construct the forms and set values if present
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.form = new this.constructor.formClass(config);
        if (!isEmpty(this.value)) {
            this.form.setValues(this.value);
        }
    }

    /**
     * Override getValue to return the values from the form
     */
    getValue() {
        return this.form.values;
    }

    /**
     * Override setValue to pass values to the form
     */
    setValue(newValue) {
        this.form.setValues(newValue);
    }

    /**
     * Pass through build() to the form
     */
    async build() {
        return await this.form.getNode();
    }
}

export { RepeatableInputView, FormInputView };
