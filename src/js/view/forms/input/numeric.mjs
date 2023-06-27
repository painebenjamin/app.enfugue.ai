/** @module view/forms/input/numeric */
import { InputView } from "./base.mjs";
import { isEmpty } from "../../../base/helpers.mjs";

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

        if (!isEmpty(this.minValue) && inputValue < this.minValue) {
            this.setValue(this.minValue, false);
        } else if (!isEmpty(this.maxValue) && inputValue > this.maxValue) {
            this.setValue(this.maxValue, false);
        } else if (!isEmpty(this.stepValue)) {
            let stepInteger = this.stepValue * Math.pow(10, this.precision),
                inputInteger = inputValue * Math.pow(10, this.precision),
                stepOffset = inputInteger % stepInteger;
            
            if (stepOffset !== 0) {
                let offsetValue = parseFloat(((inputInteger - stepOffset) / Math.pow(10, this.precision)).toFixed(this.precision));
                this.setValue(offsetValue, false);
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
    DateInputView,
    TimeInputView,
    DateTimeInputView
};
