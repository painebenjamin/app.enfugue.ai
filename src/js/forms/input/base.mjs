/** @module forms/input/base */
import { View } from "../../view/base.mjs";
import { isEmpty } from "../../base/helpers.mjs";

/**
 * The base InputView provides convenient interfaces for building
 * individual inputs or being part of a larger form.
 */
class InputView extends View {
    /**
     * @var string The tag name to render on the page.
     */
    static tagName = "input";

    /**
     * @var string The input type attribute.
     */
    static inputType = "text";

    /**
     * @var mixed The default value for the input. Optional.
     */
    static defaultValue = null;

    /**
     * @var string The placeholder for the input. Optional.
     */
    static placeholder = null;

    /**
     * @var string The value for the autocomplete attribute.
     */
    static autoComplete = "off";

    /**
     * @param object $config The overall configuration object.
     * @param string $fieldName The name of this input, should be unique to it"s context.
     * @param object $fieldConfig Configuration details for this input - see below for details.
     */
    constructor(config, fieldName, fieldConfig) {
        super(config);

        this.fieldName = fieldName;
        if (isEmpty(this.fieldName)) this.fieldName = "unnamed";
        this.fieldConfig = fieldConfig || {};

        this.value = isEmpty(this.fieldConfig.value)
            ? this.constructor.defaultValue
            : this.fieldConfig.value;

        this.errorMessage = null;
        this.autoComplete = isEmpty(this.fieldConfig.autoComplete)
            ? this.constructor.autoComplete
            : this.fieldConfig.autoComplete;
        this.editable = !(this.fieldConfig.editable === false);
        this.disabled = !!this.fieldConfig.disabled;
        this.readonly = !!this.fieldConfig.readonly;
        this.required = !!this.fieldConfig.required;
        this.placeholder = this.fieldConfig.placeholder || this.constructor.placeholder;
        this.tooltip = this.fieldConfig.tooltip || this.constructor.tooltip;

        this.onChangeCallbacks = [];
        this.onInputCallbacks = [];
        this.onFocusCallbacks = [];
        this.onBlurCallbacks = [];
        this.onClickCallbacks = [];
        this.onMouseUpCallbacks = [];
        this.onMouseDownCallbacks = [];
    }

    /**
     * Adds a callback to perform when the input is changed.
     *
     * @param callable $callback The callback function.
     */
    onChange(callback) {
        this.onChangeCallbacks.push(callback);
    }

    /**
     * Adds a callback to perform when input is performed.
     * It"s important to remember that change() only occurs after blur(),
     * this will not require leaving the field and will fire
     * when the field is changed at all.
     *
     * @param callable $callback The callback function.
     */
    onInput(callback) {
        this.onInputCallbacks.push(callback);
    }

    /**
     * Adds a callback to perform when the input is focused on.
     *
     * @param callable $callback The callback function.
     */
    onFocus(callback) {
        this.onFocusCallbacks.push(callback);
    }

    /**
     * Adds a callback to perform when the input loses focus.
     *
     * @param callable $callback The callback function.
     */
    onBlur(callback) {
        this.onBlurCallbacks.push(callback);
    }
    
    /**
     * Adds a callback to perform when the input is clicked.
     *
     * @param callable $callback The callback function.
     */
    onClick(callback) {
        this.onClickCallbacks.push(callback);
    }
    
    /**
     * Adds a callback to perform when the input first has a mouse down.
     *
     * @param callable $callback The callback function.
     */
    onMouseDown(callback) {
        this.onMouseDownCallbacks.push(callback);
    }
    
    /**
     * Adds a callback to perform when the input first has a mouse up.
     *
     * @param callable $callback The callback function.
     */
    onMouseUp(callback) {
        this.onMouseUpCallbacks.push(callback);
    }

    /**
     * Force focus on this input.
     */
    focus() {
        if (this.node !== undefined) {
            this.node.focus();
        }
    }

    /**
     * Trigger onChange callbacks.
     *
     * @param Event $e The change event. Optional.
     */
    changed(e) {
        if (e) {
            e.stopPropagation();
            e.preventDefault();
        }
        let lastValue = this.value,
            newValue = this.getValue();

        for (let onChangeCallback of this.onChangeCallbacks) {
            onChangeCallback(this.fieldName, lastValue, newValue);
        }

        this.value = newValue;
    }

    /**
     * Trigger onInput callbacks.
     *
     * @param Event $e The input event.
     */
    inputted(e) {
        if (e) {
            e.stopPropagation();
            e.preventDefault();
        }
        let currentValue = this.getValue();
        for (let inputCallback of this.onInputCallbacks) {
            inputCallback(currentValue);
        }
    }
    
    /**
     * Trigger onFocus callbacks.
     *
     * @param Event $e The focus event.
     */
    focused(e) {
        if (e) {
            e.stopPropagation();
            e.preventDefault();
        }
        for (let focusCallback of this.onFocusCallbacks) {
            focusCallback(e);
        }
    }

    /**
     * Trigger onBlur callbacks.
     *
     * @param Event $e The blur event.
     */
    blurred(e) {
        if (e) {
            e.stopPropagation();
            e.preventDefault();
        }
        for (let blurCallback of this.onBlurCallbacks) {
            blurCallback(e);
        }
    }
    
    /**
     * Trigger onClick callbacks.
     *
     * @param Event $e The click event.
     */
    clicked(e) {
        if (e) {
            e.stopPropagation();
        }
        for (let clickCallback of this.onClickCallbacks) {
            clickCallback(e);
        }
    }
    
    /**
     * Trigger onMouseUp callbacks.
     *
     * @param Event $e The mouseup event.
     */
    mouseUpped(e) {
        if (e) {
            e.stopPropagation();
        }
        for (let mouseUpCallback of this.onMouseUpCallbacks) {
            mouseUpCallback(e);
        }
    }
    
    /**
     * Trigger onMouseDown callbacks.
     *
     * @param Event $e The mouseup event.
     */
    mouseDowned(e) {
        if (e) {
            e.stopPropagation();
        }
        for (let mouseDownCallback of this.onMouseDownCallbacks) {
            mouseDownCallback(e);
        }
    }

    /**
     * Disables this input.
     */
    disable() {
        this.disabled = true;
        if (this.node !== undefined) {
            this.node.disabled(true);
        }
        return this;
    }

    /**
     * Enables a field, but only if it isn't read-only.
     */
    checkEnable() {
        if (!this.editable && !isEmpty(this.value)) {
            return this;
        }
        return this.enable();
    }

    /**
     * Enables a field.
     */
    enable() {
        this.disabled = false;
        if (this.node !== undefined) {
            this.node.disabled(false);
        }
        return this;
    }

    /**
     * Sets the value of the field.
     *
     * @param mixed $newValue The new value to set in this field.
     * @param bool $triggerChange Whether or not to trigger the change callbacks. Default true.
     */
    setValue(newValue, triggerChange) {
        let valueChanged = this.value !== newValue;
        this.value = newValue;

        if (this.node !== undefined) {
            this.node.val(this.value, false);
        }

        if (triggerChange !== false && valueChanged) {
            this.changed();
        }

        if (!this.editable) {
            this.disable();
        }
        return this;
    }

    /**
     * Get the value currently set.
     */
    getValue() {
        if (this.node !== undefined && this.node.element !== undefined) {
            return this.node.val();
        }
        return this.value;
    }

    /**
     * Get the value currently set. If this is required and there is no value,
     * throw an exception.
     *
     * @param bool $enforceRequired Whether or not to enforce required-ness. Default true.
     */
    checkGetValue(enforceRequired) {
        let value = this.getValue();

        if (this.required && isEmpty(value) && enforceRequired !== false) {
            throw "This field is required.";
        }

        return value;
    }

    /**
     * Triggered by a form when the label is clicked.
     */
    labelClicked() {
        this.focus();
    }

    /**
     * Builds the inputs element.
     */
    async build() {
        let node = await super.build();
        node.name(this.fieldName)
            .attr("disabled", this.disabled)
            .attr("readonly", this.readonly)
            .attr("required", this.required)
            .attr("autocomplete", this.autoComplete)
            .on("change", (e) => this.changed(e))
            .on("input", (e) => this.inputted(e))
            .on("blur", (e) => this.blurred(e))
            .on("focus", (e) => this.focused(e))
            .on("click", (e) => this.clicked(e))
            .on("mouseup", (e) => this.mouseUpped(e))
            .on("mousedown", (e) => this.mouseDowned(e));

        if (this.constructor.tagName == "input") {
            node.type(this.constructor.inputType);
        }

        if (!isEmpty(this.value)) {
            node.val(this.value);
        }

        if (!isEmpty(this.placeholder)) {
            node.attr("placeholder", this.placeholder);
        }

        if (!isEmpty(this.tooltip)) {
            node.data("tooltip", this.tooltip);
        }

        return node;
    }
}

export { InputView };
