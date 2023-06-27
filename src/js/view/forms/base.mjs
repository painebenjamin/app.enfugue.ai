/** @module view/forms/base */
import { View } from '../base.mjs';
import { ElementBuilder } from '../../base/builder.mjs';
import { isEmpty, kebabCase, set } from '../../base/helpers.mjs';

const E = new ElementBuilder();

/**
 * The FormView provides an easily extensible form class.
 * It can be extended in code or at runtime.
 */
class FormView extends View {
    /**
     * @var string The tag name of the element built
     */
    static tagName = 'form';

    /**
     * @var object A key-value map of field set configuration
     */
    static fieldSets = {};

    /**
     * @var object A key-value map of field set display conditions
     */
    static fieldSetConditions = {};

    /**
     * @var bool Whether or not to submit when any changes are made and hide the submit button.
     */
    static autoSubmit = false;

    /**
     * @var bool Remove the submit button, but don't auto submit. Default false.
     */
    static noSubmit = false;

    /**
     * @var bool Whether or not to disable the form on submissions. Default true.
     */
    static disableOnSubmit = true;

    /**
     * @var string The autocomplete variable controls whether or not browsers will try and autofill.
     */
    static autoComplete = 'off';

    /**
     * @var string The label to fit in the submit button
     */
    static submitLabel = 'Submit';

    /**
     * @var bool Whether or not to show the 'cancel' button. Default false.
     */
    static showCancel = false;

    /**
     * @var text The text to show in the 'cancel' button.
     */
    static cancelLabel = 'Cancel';

    /**
     * @var bool Whether or not each field set should be shown collapsed instead of always visible
     */
    static collapseFieldSets = false;

    /**
     * @param object $config The configuration object.
     * @param object $values An optional initial key-value map of values.
     */
    constructor(config, values) {
        super(config);
        this.values = values || {};
        this.errors = {};
        this.submitCallbacks = [];
        this.cancelCallbacks = [];
        this.changeCallbacks = [];
        this.inputViews = [];
        this.disabled = false;
        this.canceled = false;
        this.dynamicFieldSets = {};
    }

    /**
     * Sets the forms values.
     *
     * @param object $newValues A key-value map of new values.
     */
    async setValues(newValues) {
        await this.getNode(); // Wait for build
        this.values = Object.getOwnPropertyNames(newValues).reduce((accumulator, key) => {
            if (this.inputViews.map((inputView) => inputView.fieldName).indexOf(key) !== -1) {
                accumulator[key] = newValues[key];
            }
            return accumulator;
        }, {});
        await Promise.all(
            this.inputViews.map((inputView) => 
                inputView.setValue(this.values[inputView.fieldName], false)
            )
        );
        await this.evaluateConditions();
        for (let inputView of this.inputViews) {
            this.values[inputView.fieldName] = inputView.getValue();
        }
        return this;
    }

    /**
     * Gets the input view generated for a field.
     *
     * @param string $fieldName The field to get.
     * @return InputView
     */
    async getInputView(fieldName) {
        if (this.node === undefined) {
            await this.getNode();
        }
        return this.inputViews
            .filter((inputView) => inputView.fieldName === fieldName)
            .shift();
    }

    /**
     * Adds a new input after instantiation.
     *
     * @param string $fieldSetName The name of the field set, generally is displayed.
     * @param class $fieldClass The input class to instantiate.
     * @param string $fieldName The unique name of this field
     * @param string $fieldLabel The label to shown with this field
     * @param object $fieldConfig Configuration to pass to the constuctor.
     * @return InputView The constructed instance of $fieldClass
     */
    async addInput(
        fieldSetName,
        fieldClass,
        fieldName,
        fieldLabel,
        fieldConfig
    ) {
        let inputView = new fieldClass(this.config, fieldName, fieldConfig);

        if (this.dynamicFieldSets[fieldSetName] === undefined) {
            this.dynamicFieldSets[fieldSetName] = {};
        }

        this.dynamicFieldSets[fieldSetName][fieldName] = {
            instance: inputView,
            label: fieldLabel,
            class: fieldClass,
            config: fieldConfig
        };

        inputView.onChange(() => this.inputChanged(fieldName, inputView));
        this.inputViews.push(inputView);

        if (this.node !== undefined) {
            let fieldSet = this.node.find(
                    `fieldset.field-set-${kebabCase(fieldSetName)}`
                ),
                labelNode = E.label().content(fieldLabel).for(fieldName),
                errorNode = E.p().class('error').for(fieldName);

            if (fieldSet === null) {
                fieldSet = E.fieldset()
                    .content(E.legend().content(fieldSetName))
                    .class(`field-set-${kebabCase(fieldSetName)}`);
                this.node.append(fieldSet);
            }

            inputView.fieldSet = fieldSet;

            if (!isEmpty(this.errors[fieldName])) {
                errorNode.content(this.errors[fieldName]);
                labelNode.addClass('error');
            }

            if (inputView.required) {
                labelNode.addClass('required');
            }

            if (!isEmpty(inputView.tooltip)) {
                labelNode.data("tooltip", inputView.tooltip);
            } else if (!isEmpty(fieldConfig.tooltip)) {
                labelNode.data("tooltip", fieldConfig.tooltip);
            }

            if (isEmpty(fieldLabel)) {
                labelNode.hide();
            }

            if (this.values.hasOwnProperty(fieldName)) {
                inputView.setValue(this.values[fieldName], false);
            }

            inputView.onChange(() => this.inputChanged(fieldName, inputView));

            this.inputViews.push(inputView);
            let fieldContainer = E.div()
                .class('field-container')
                .content(errorNode, await inputView.getNode(), labelNode);
            fieldContainer.addClass(kebabCase(inputView.constructor.name));
            fieldSet.append(fieldContainer);
        }

        return inputView;
    }

    /**
     * Disables the form and all inputs.
     */
    disable() {
        this.disabled = true;
        for (let inputView of this.inputViews) {
            inputView.disable();
        }
        if (
            !this.constructor.autoSubmit &&
            !this.constructor.noSubmit &&
            this.node !== undefined
        ) {
            for (let node of this.node.findAll("input.submit")) {
                node.disabled(true);
            }
        }
    }

    /**
     * Enables the form and all inputs that should be re-enabled.
     */
    enable() {
        this.disabled = false;
        for (let inputView of this.inputViews) {
            inputView.checkEnable();
        }
        if (
            !this.constructor.autoSubmit &&
            !this.constructor.noSubmit &&
            this.node !== undefined
        ) {
            for (let node of this.node.findAll("input.submit")) {
                node.disabled(false);
            }
        }
    }

    /**
     * Removes the error message.
     */
    clearError() {
        this.setError('');
    }

    /**
     * Sets the error message.
     *
     * @param mixed $errorMessage The error message to set. Could be a bad XHR, string, or exception.
     */
    setError(errorMessage) {
        if (errorMessage instanceof XMLHttpRequest) {
            try {
                errorMessage = JSON.parse(errorMessage.responseText);
            } catch (e) {
                errorMessage = errorMessage.toString();
            }
        }
        if (typeof errorMessage !== 'string') {
            if (errorMessage.errors !== undefined) {
                errorMessage = errorMessage.errors[0];
            }
            if (errorMessage.detail !== undefined) {
                errorMessage = `${errorMessage.title}: ${errorMessage.detail}`;
            } else {
                console.error(errorMessage);
                errorMessage = errorMessage.toString();
            }
        }
        this.errorMessage = errorMessage;
        if (this.node !== undefined) {
            if (isEmpty(errorMessage)) {
                this.node.find('p.error').empty().hide();
            } else {
                this.node.find('p.error').content(errorMessage).show();
            }
        }
    }

    /**
     * Adds a callback to fire when any field is changed.
     *
     * @param callable $callback The function to call with the new values when data is changed.
     */
    onChange(callback) {
        this.changeCallbacks.push(callback);
    }

    /**
     * Adds a callback to fire when the form is submitted.
     * If this has `autoSubmit` = true, this functions the same as `onChange`, except it will only
     * be called if necessary conditions are met based on the input.
     *
     * @param callable $callback The function to call with the new values when the form is submitted.
     */
    onSubmit(callback) {
        this.submitCallbacks.push(callback);
    }

    /**
     * Adds a callback to fire when the form is cancelled.
     * This is not shown by default.
     */
    onCancel(callback) {
        this.cancelCallbacks.push(callback);
    }

    /**
     * Cancels the form.
     */
    async cancel() {
        for (let cancelCallback of this.cancelCallbacks) {
            await cancelCallback();
        }
        this.disabled = true;
        this.canceled = true;
    }

    /**
     * Submits the form.
     */
    async submit() {
        if (this.disabled) {
            throw 'Form is disabled.';
        }

        this.addClass('loading');

        let isError = false,
            conditions = {};

        for (let inputView of this.inputViews) {
            let errorTextNode, inputLabelNode;

            if (this.node !== undefined) {
                errorTextNode = this.node.find(
                    `p.error[data-for='${inputView.node.id()}']`
                );
                inputLabelNode = this.node.find(
                    `label[data-for='${inputView.node.id()}']`
                );
            }

            try {
                let enforceRequired = true;
                if (
                    !isEmpty(
                        this.constructor.fieldSetConditions[inputView.fieldSet]
                    )
                ) {
                    if (isEmpty(conditions[inputView.fieldSet])) {
                        conditions[inputView.fieldSet] =
                            this.constructor.fieldSetConditions[
                                inputView.fieldSet
                            ](this.values);
                    }
                    enforceRequired = conditions[inputView.fieldSet];
                }
                this.values[inputView.fieldName] = inputView.checkGetValue(enforceRequired);
                if (this.node !== undefined) {
                    if (!isEmpty(errorTextNode)) {
                        errorTextNode.empty();
                    }
                    if (!isEmpty(inputLabelNode)) {
                        inputLabelNode.removeClass('error');
                    }
                }
            } catch (e) {
                isError = true;
                this.errors[inputView.fieldName] = e;
                if (this.node !== undefined) {
                    if (!isEmpty(errorTextNode)) {
                        errorTextNode.content(e);
                    }
                    if (!isEmpty(inputLabelNode)) {
                        inputLabelNode.addClass('error');
                    }
                }
            }
        }

        if (isError) {
            this.setError('Error');
            this.removeClass('loading');
            return;
        }

        if (this.constructor.disableOnSubmit && !this.constructor.autoSubmit) {
            this.disable();
        }

        this.submitResults = [];
        for (let callback of this.submitCallbacks) {
            try {
                this.submitResults.push(await callback(this.values));
            } catch (e) {
                this.setError(e);
                this.enable();
                break;
            }
        }
        this.removeClass('loading');
    }

    /**
     * This function is called to re-evaluate visibility conditions of field sets
     * when an input is changed.
     */
    async evaluateConditions() {
        if (this.node !== undefined) {
            for (let fieldSet in this.constructor.fieldSetConditions) {
                let fieldSetNode = this.node.find(`fieldset.field-set-${kebabCase(fieldSet)}`);
                if (
                    this.constructor.fieldSetConditions[fieldSet](this.values)
                ) {
                    if (!isEmpty(fieldSetNode)) {
                        fieldSetNode.show();
                    }
                    for (let inputView of this.inputViews) {
                        if (
                            inputView.required &&
                            inputView.fieldSet == fieldSet
                        ) {
                            (await inputView.getNode()).attr('required', true);
                        }
                    }
                } else {
                    if (!isEmpty(fieldSetNode)) {
                        fieldSetNode.hide();
                    }
                    for (let inputView of this.inputViews) {
                        if (
                            inputView.required &&
                            inputView.fieldSet == fieldSet
                        ) {
                            (await inputView.getNode()).attr('required', false);
                        }
                    }
                }
            }
        }
    }

    /**
     * This function is called when an individual input is changed and triggers
     * all of the necessary callbacks and conditions.
     *
     * @param string $fieldName The name of the field.
     * @param InputView $inputView The input view that changed.
     */
    async inputChanged(fieldName, inputView) {
        this.values[fieldName] = inputView.getValue();

        if (this.node !== undefined) {
            await this.evaluateConditions();
        }

        for (let callback of this.changeCallbacks) {
            await callback(fieldName, this.values);
        }

        if (this.constructor.autoSubmit) {
            await this.submit();
        }
    }

    /**
     * On build, read configuration and generate views.
     */
    async build() {
        let node = await super.build(),
            fieldSets = this.constructor.fieldSets,
            dynamicFieldSets = this.dynamicFieldSets,
            allFieldSets = set(
                Object.getOwnPropertyNames(fieldSets).concat(
                    Object.getOwnPropertyNames(dynamicFieldSets)
                )
            );

        node.attr('autocomplete', this.constructor.autoComplete);

        for (let fieldSet of allFieldSets) {
            let legendNode = E.legend().content(fieldSet),
                fieldSetNode = E.fieldset()
                    .content(legendNode)
                    .class(`field-set-${kebabCase(fieldSet)}`);

            let fieldSetCondition =
                    this.constructor.fieldSetConditions[fieldSet],
                fieldSetIsHidden = false,
                isDynamic = dynamicFieldSets[fieldSet] !== undefined,
                fieldSetConfig = isDynamic
                    ? dynamicFieldSets[fieldSet]
                    : fieldSets[fieldSet];

            if (!isEmpty(fieldSetCondition)) {
                fieldSetIsHidden = !fieldSetCondition(this.values);
            }

            if (fieldSetIsHidden) {
                fieldSetNode.hide();
            }

            if (this.constructor.collapseFieldSets) {
                fieldSetNode.addClass("collapsible collapsed");
                legendNode.on("click", (e) => {
                    e.stopPropagation();
                    fieldSetNode.toggleClass("collapsed");
                });
            }

            for (let fieldName in fieldSetConfig) {
                let fieldContainer = E.div().class('field-container'),
                    fieldClass = fieldSetConfig[fieldName].class,
                    fieldInstance = fieldSetConfig[fieldName].instance,
                    label = fieldSetConfig[fieldName].label,
                    config = fieldSetConfig[fieldName].config,
                    fieldId = fieldContainer.elementId; // This is a UUID

                if (isEmpty(fieldClass)) throw `Field ${fieldName} does not provide a class.`;

                if (isEmpty(config)) {
                    config = {};
                }

                let labelNode = E.label().content(label).data('for', fieldId),
                    errorNode = E.p().class('error').data('for', fieldId),
                    inputView;

                if (isDynamic) {
                    inputView = fieldInstance;
                    fieldContainer.addClass(
                        kebabCase(fieldInstance.constructor.name)
                    );
                    fieldInstance.formParent = this;
                } else {
                    fieldContainer.addClass(kebabCase(fieldClass.name));
                    inputView = new fieldClass(this.config, fieldName, config);
                    inputView.onChange(() =>
                        this.inputChanged(fieldName, inputView)
                    );
                    this.inputViews.push(inputView);
                    inputView.formParent = this;
                }

                inputView.fieldSet = fieldSet;

                if (!isEmpty(this.errors[fieldName])) {
                    errorNode.content(this.errors[fieldName]);
                    labelNode.addClass('error');
                }

                if (inputView.required) {
                    labelNode.addClass('required');
                }

                if (!isEmpty(inputView.tooltip)) {
                    labelNode.data("tooltip", inputView.tooltip);
                }

                if (isEmpty(label)) {
                    labelNode.hide();
                }

                if (this.values.hasOwnProperty(fieldName)) {
                    inputView.setValue(this.values[fieldName]);
                } else {
                    let defaultValue = inputView.getValue();
                    if (defaultValue) {
                        this.values[fieldName] = defaultValue;
                    }
                }

                labelNode.on("click", () => inputView.labelClicked());

                fieldSetNode.append(
                    fieldContainer.content(
                        errorNode,
                        (await inputView.getNode()).id(fieldId),
                        labelNode
                    )
                );

                if (inputView.required && fieldSetIsHidden) {
                    inputView.node.attr('required', false);
                }
            }
            node.append(fieldSetNode);
        }

        let errorMessage = E.p().class('error');

        if (isEmpty(this.errorMessage)) {
            errorMessage.hide();
        } else {
            errorMessage.content(this.errorMessage);
        }

        node.append(errorMessage);


        if (!this.constructor.autoSubmit && !this.constructor.noSubmit) {
            let submitButton = E.input().type('submit')
                    .class('submit')
                    .value(this.constructor.submitLabel),
                submitContainer = E.div().class('submit-buttons')
                    .content(submitButton);

            node.append(submitContainer);

            if (this.constructor.showCancel) {
                node.addClass("can-cancel");
                let cancelButton = E.input()
                    .type('button')
                    .class('submit cancel')
                    .value(this.constructor.cancelLabel);

                cancelButton.on("click", () => { 
                    if (!cancelButton.hasClass("disabled")){
                        this.cancel(); 
                    }
                });
                submitContainer.append(cancelButton);
            }
        }

        node.on('submit', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.submit();
        });

        return node;
    }
}

export { FormView };
