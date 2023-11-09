/** @module forms/input/numeric */
import { InputView } from "./base.mjs";
import { isEmpty } from "../../base/helpers.mjs";

/**
 * Extends the InputView to set type to number
 */
class NumberInputView extends InputView {
    /**
     * @var int The minimum value
     */
    static min = null;

    /**
     * @var int The maximum value
     */
    static max = null;

    /**
     * @var int The step change when clicking up or down
     */
    static step = 1;

    /**
     * @var string The input type
     */
    static inputType = "number";

    /**
     * @var bool Whether or not to bind mousewheel input, default true
     */
    static bindMouseWheel = true;

    /**
     * @var bool Whether or not to require the alt key when scolling the mousewheel, default true
     */
    static useMouseWheelAltKey = true;

    /**
     * @var Whether or not to allow null values
     */
    static allowNull = true;

    /**
     * The constructor just sets static values to mutable local ones.
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.minValue = isEmpty(fieldConfig) || isEmpty(fieldConfig.min)
            ? this.constructor.min
            : fieldConfig.min;
        this.maxValue = isEmpty(fieldConfig) || isEmpty(fieldConfig.max)
            ? this.constructor.max
            : fieldConfig.max;
        this.stepValue = isEmpty(fieldConfig) || isEmpty(fieldConfig.step)
            ? this.constructor.step
            : fieldConfig.step;
        this.allowNull = isEmpty(fieldConfig) || isEmpty(fieldConfig.allowNull)
            ? this.constructor.allowNull
            : fieldConfig.allowNull;
    }

    /**
     * @return int The precision (based on step)
     */
    get precision() {
        if (isEmpty(this.stepValue) || this.stepValue >= 1 || this.stepValue < 0) {
            return 0;
        }
        // Floating point
        let stepParts = `${this.stepValue}`.split(".");
        if (stepParts.length < 2) {
            return 0;
        }
        return stepParts[1].length;
    }

    /**
     * @param int $newMin The new minimum value
     */
    setMin(newMin) {
        this.minValue = newMin;
        if (this.node !== undefined){
            this.node.attr("min", this.minValue);
        }
    }

    /**
     * @param int $newMax The new maximum value
     */
    setMax(newMax) {
        this.maxValue = newMax;
        if (this.node !== undefined){
            this.node.attr("max", this.maxValue);
        }
    }
    
    /**
     * @param int $newStep The new step value
     */
    setStep(newStep) {
        this.stepValue = newStep;
        if (this.node !== undefined){
            this.node.attr("step", this.stepValue);
        }
    }

    /**
     * Checks min/max/step on the value and alters it if needed
     */
    checkValue() {
        let inputValue = this.getValue(),
            lastValue = this.value;

        if (isEmpty(inputValue) || isNaN(inputValue)) {
            if (this.allowNull) return;
            inputValue = isEmpty(this.minValue) ? 0 : this.minValue;
            this.setValue(inputValue, false);
        }

        if (!isEmpty(this.minValue) && inputValue < this.minValue) {
            this.setValue(this.minValue, false);
        } else if (!isEmpty(this.maxValue) && inputValue > this.maxValue) {
            this.setValue(this.maxValue, false);
        } else if (!isEmpty(this.stepValue)) {
            let stepInteger = Math.round(this.stepValue * Math.pow(10, this.precision)),
                inputInteger = Math.round(inputValue * Math.pow(10, this.precision)),
                stepOffset = inputInteger % stepInteger;
            if (stepOffset !== 0) {
                let offsetValue = parseFloat(((inputInteger - stepOffset) / Math.pow(10, this.precision)).toFixed(this.precision));
                this.setValue(offsetValue, false);
            } else {
                let recalculatedValue = inputInteger / Math.pow(10, this.precision);
                if (recalculatedValue != inputValue) {
                    this.setValue(recalculatedValue, false);
                }
            }
        }
    }

    /**
     * When changed, check the value for min/max/step.
     */
    changed() {
        this.checkValue();
        super.changed();
    }
    
    /**
     * Gets the value as numeric
     *
     * @return numeric The value of the input
     */
    getValue(value) {
        let inputValue = super.getValue(value);
        if (isEmpty(this.stepValue) || this.stepValue >= 1) {
            return parseInt(inputValue);
        }
        return parseFloat(inputValue);
    }

    /**
     * On build, set necessary attributes.
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.minValue)) {
            node.attr("min", this.minValue);
        }
        if (!isEmpty(this.maxValue)) {
            node.attr("max", this.maxValue);
        }
        if (!isEmpty(this.stepValue)) {
            node.attr("step", this.stepValue);
        }
        if (this.constructor.bindMouseWheel) {
            node.on("mousewheel", (e) => {
                if (this.constructor.useMouseWheelAltKey && !e.altKey) return;
                e.preventDefault();
                e.stopPropagation();
                let toAdd = isEmpty(this.stepValue) ? 1 : this.stepValue;
                if (e.deltaY > 0) {
                    toAdd *= -1;
                }
                this.setValue(this.value + toAdd);
            });
        }
        return node;
    }
}

/**
 * This isn't an actual input type, but symbolically allows any level of precision
 */
class FloatInputView extends NumberInputView {
    /**
     * @var string The step value, set the 'any' to disable step checking
     */
    static step = "any";
}

/**
 * Extends the number type to use slider
 */
class SliderInputView extends NumberInputView {
    /**
     * @var string The input type
     */
    static inputType = "range";
}

/**
 * Combines a slider and number input.
 */
class SliderPreciseInputView extends NumberInputView {
    /**
     * @var string The custom tag name for this input
     */
    static tagName = "enfugue-slider-precise-input-view";

    /**
     * @var class The class for the numberic precise input.
     */
    static numberInputClass = NumberInputView;

    /**
     * @var class The class for the slider input.
     */
    static sliderInputClass = SliderInputView;

    /**
     * The constructor passes the arguments to both child classes.
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        for (let keyName of ["min", "max", "step"]) {
            if (isEmpty(fieldConfig[keyName])) {
                fieldConfig[keyName] = this.constructor[keyName];
            }
        }
        if (isEmpty(fieldConfig.value)) {
            fieldConfig.value = this.constructor.defaultValue;
        }
        this.sliderInput = new this.constructor.sliderInputClass(config, fieldName, fieldConfig);
        this.numberInput = new this.constructor.numberInputClass(config, `${fieldName}Precise`, fieldConfig);

        this.sliderInput.onInput((value) => {
            this.value = value;
            this.numberInput.setValue(value, false);
            this.changed();
        });

        this.numberInput.onInput((value) => {
            this.value = value;
            this.sliderInput.setValue(value, false);
            this.changed();
        });
    }

    /**
     * When disabling, pass to both inputs.
     */
    disable() {
        super.disable();
        this.sliderInput.disable();
        this.numberInput.disable();
    }

    /**
     * When enabling, pass to both inputs.
     */
    enable() {
        super.enable();
        this.sliderInput.enable();
        this.numberInput.enable();
    }

    /**
     * Always return the value from memory.
     */
    getValue() {
        return this.value;
    }

    /**
     * When setting value, set both child inputs.
     */
    setValue(value, triggerChange) {
        let result = super.setValue(value, false);
        this.sliderInput.setValue(value, false);
        this.numberInput.setValue(value, false);
        if (triggerChange) {
            this.changed();
        }
    }

    /**
     * On build, get both child input nodes.
     */
    async build() {
        let node = await super.build();
        node.append(
            await this.sliderInput.getNode(),
            await this.numberInput.getNode()
        );
        return node;
    }
};

/**
 * This class uses the base datetime-local type, but parses values as needed.
 */
class DateTimeInputView extends NumberInputView {
    /**
     * @var string The input type
     */
    static inputType = "datetime-local";

    /**
     * @var int Override the step value
     */
    static step = null;

    /**
     * Allow for Date values
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        if (this.value instanceof Date) {
            this.value = this.value.toISOString().split(".")[0];
        }
    }

    /**
     * Set the value
     *
     * @param Date value The new value
     */
    setValue(value) {
        super.setValue(value);
        if (this.value instanceof Date) {
            this.value = this.value.toISOString().split(".")[0];
        }
    }

    /**
     * Gets the value as a Date
     *
     * @return Date The value of the input
     */
    getValue(value) {
        return new Date(this.value);
    }
}

/**
 * This is like DateTime, but doesn"t have a time.
 */
class DateInputView extends NumberInputView {
    /**
     * @var string The input type
     */
    static inputType = "date";

    /**
     * @var int Override the step value
     */
    static step = null;

    /**
     * Allow for Date values
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        if (this.value instanceof Date) {
            this.value = this.value.toISOString().split("T")[0];
        }
    }

    /**
     * Sets the value, allows Date types
     *
     * @param Date $value The value to set
     */
    setValue(value) {
        if (value instanceof Date) {
            try {
                value = value.toISOString().split("T")[0];
            } catch (e) {
                value = null; // "Invalid Date"
            }
        }
        super.setValue(value);
    }

    /**
     * Gets the value as a Date
     *
     * @return Date the value opf the input
     */
    getValue(value) {
        if (this.node !== undefined) {
            return new Date(this.node.val());
        }
        return new Date(this.value);
    }
}

/**
 * The TimeInputView uses the normal Time input type.
 */
class TimeInputView extends NumberInputView {
    /**
     * @var int Override the step value
     */
    static step = null;

    /**
     * @var string The input type
     */
    static inputType = "time";
}

export {
    NumberInputView,
    FloatInputView,
    SliderInputView,
    SliderPreciseInputView,
    DateInputView,
    TimeInputView,
    DateTimeInputView
};
