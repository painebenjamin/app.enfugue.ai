import { View } from "../../base.mjs";
import { ElementBuilder } from "../../../base/builder.mjs";
import { InputView } from "./base.mjs";
import { StringInputView } from "./string.mjs";
import {
    isEmpty,
    hslToHex,
    hslToRgb,
    hexToHsl
} from "../../../base/helpers.mjs";

const E = new ElementBuilder();

/**
 * This view shows the popup color selector part of the ColorInputView.
 */
class ColorInputHelperView extends View {
    /**
     * @var string The tag name of the helper node
     */
    static tagName = "enfugue-color-input-helper-view";

    /**
     * @param object $config the main configuration object.
     * @param int $h Optional the hue from 0-360
     * @param float $s Optional, the saturation from 0-1
     * @param float $l Optionalo, the lightness from 0-1
     */
    constructor(config, h, s, l) {
        super(config);
        this.h = isEmpty(h) ? 0 : h * 360;
        this.s = isEmpty(s) ? 1.0 : s;
        this.l = isEmpty(l) ? 0.5 : l;
        this.changeCallbacks = [];
    }

    /**
     * We have to use callbacks in memory since this is a non-standard input
     *
     * @param callable $callback The function to call when this input changes.
     */
    onChange(callback) {
        this.changeCallbacks.push(callback);
    }

    /**
     * @return string The color value as a hexadecimal string.
     */
    get hex() {
        return hslToHex(this.h / 360, this.s, this.l);
    }

    /**
     * @return array<int> The color value as three 0-255 integers (RGB)
     */
    get rgb() {
        return hslToRgb(this.h / 360, this.s, this.l);
    }

    /**
     * @return array<float> The color vlaue as three 0-1 floats (HSL)
     */
    get hsl() {
        return [this.h / 360, this.s, this.l];
    }

    /**
     * Sets the value by passing a hex value.
     *
     * @param string $newHex The new hex value to set.
     */
    set hex(newHex) {
        if (!isEmpty(newHex)) {
            try {
                let [h, s, l] = hexToHsl(newHex);
                this.h = h * 360;
                this.s = s;
                this.l = l;
            } catch (e) {
                console.warn("Couldn't parse hexadecimal value", newHex);
            }
        }
        this.checkUpdateNode();
    }

    /**
     * Setter for the hue value.
     *
     * @param int $newHue The hue from 0-360.
     */
    set hue(newHue) {
        this.h = newHue;
        this.checkUpdateNode();
    }

    /**
     * @return int The hue from 0-360
     */
    get hue() {
        return this.h;
    }

    /**
     * Setter for the saturation value.
     *
     * @param float $newSaturation The new saturation value, 0-1.
     */
    set saturation(newSaturation) {
        this.s = newSaturation;
        this.checkUpdateNode();
    }

    /**
     * @return string The saturation as percentage string
     */
    get saturation() {
        return `${this.s * 100}%`;
    }

    /**
     * Setter for the lightness value
     *
     * @param float $newLightness The new lightness value, 0-1
     */
    set lightness(newLightness) {
        this.l = newLightness;
        this.checkUpdateNode();
    }

    /**
     * @return string The lightness value as a percentage string
     */
    get lightness() {
        return `${this.l * 100}%`;
    }

    /**
     * @return string The CSS background for the hue bar.
     */
    get hueBackground() {
        let colors = new Array(360)
            .fill(null)
            .map((_, i) => `hsl(${i}, 100%, 50%)`)
            .join(", ");
        
        return `linear-gradient(to right, ${colors})`;
    }

    /**
     * @return string The CSS background for the saturation bar.
     */
    get saturationBackground() {
        let colors = new Array(100)
                .fill(null)
                .map((_, i) => `hsl(${this.hue}, ${i}%, 50%)`)
                .join(", ");

        return `linear-gradient(to right, ${colors})`;
    }

    /**
     * @return string The CSS background for the lightness bar.
     */
    get lightnessBackground() {
        let colors = new Array(100)
                .fill(null)
                .map((_, i) => `hsl(${this.hue}, ${this.saturation}, ${i}%)`)
                .join(", ");

        return `linear-gradient(to right, ${colors})`;
    }

    /**
     * If this node is on the page, makes sure that it is up-to-date
     * with the values stored in memory.
     */
    async checkUpdateNode() {
        if (this.node !== undefined) {
            let inputParts = this.node.findAll(".input-part"),
                inputIndicators = inputParts.map((inputPart) => inputPart.find(".indicator")),
                [hueInputContainer, saturationInputContainer, lightnessInputContainer] = inputParts,
                [hueIndicator, saturationIndicator, lightnessIndicator] = inputIndicators,
                preview = this.node.find(".preview");

            hueIndicator.css("left", `${(this.h / 360) * 100}%`);
            saturationIndicator.css("left", `${this.s * 100}%`);
            lightnessIndicator.css("left", `${this.l * 100}%`);

            hueInputContainer.css("background-image", this.hueBackground);
            saturationInputContainer.css("background-image", this.saturationBackground);
            lightnessInputContainer.css("background-image", this.lightnessBackground);

            preview.css("background-color", this.hex);
        }
    }

    /**
     * Triggers change callbacks.
     */
    async changed() {
        for (let callback of this.changeCallbacks) {
            await callback(this.hex);
        }
    }

    /**
     * Builds the node and binds events.
     */
    async build() {
        let node = await super.build(),
            hueIndicator = E.div().class("indicator"),
            hueInputContainer = E.div()
                .class("input-part")
                .content(hueIndicator),
            saturationIndicator = E.div().class("indicator"),
            saturationInputContainer = E.div()
                .class("input-part")
                .content(saturationIndicator),
            lightnessIndicator = E.div().class("indicator"),
            lightnessInputContainer = E.div()
                .class("input-part")
                .content(lightnessIndicator),
            inputContainer = E.div()
                .class("input-container")
                .content(
                    hueInputContainer,
                    saturationInputContainer,
                    lightnessInputContainer
                ),
            preview = E.div().class("preview"),
            previewContainer = E.div()
                .class("preview-input-container")
                .content(preview, inputContainer);

        node.on("mouseenter", (e) => {
            this.within = true;
        }).on("mouseleave", (e) => {
            this.within = false;
            if (this.hideOnLeave) {
                this.hideOnLeave = false;
                node.hide();
            }
        });

        hueIndicator.css("left", `${this.h / 360}$`);
        saturationIndicator.css("left", `${this.s * 100}%`);
        lightnessIndicator.css("left", `${this.l * 100}%`);

        hueInputContainer.css("background-image", this.hueBackground);
        saturationInputContainer.css("background-image", this.saturationBackground);
        lightnessInputContainer.css("background-image", this.lightnessBackground);

        preview
            .css("background-color", this.hex)
            .content(
                E.div().css("color", "white").content("Preview"),
                E.div().css("color", "black").content("Preview")
            );

        let bindHandler = (container, indicator, callback) => {
            let ratio,
                getRatio = (e) => {
                    let containerBox =
                            container.element.getBoundingClientRect(),
                        containerRelativePosition = e.clientX - containerBox.x;
                    return containerRelativePosition / containerBox.width;
                },
                setRatio = (newRatio) => {
                    newRatio = Math.max(0, Math.min(newRatio, 1));
                    callback(newRatio, ratio);
                    ratio = newRatio;
                    this.checkUpdateNode();
                    this.changed();
                };

            container
                .on("mousedown", (e) => {
                    setRatio(getRatio(e));
                    container
                        .on("mousemove", (e2) => {
                            setRatio(getRatio(e2));
                        })
                        .on("mouseup,mouseleave", (e2) => {
                            e2.preventDefault();
                            e2.stopPropagation();
                            setRatio(getRatio(e2));
                            container.off("mouseup,mouseleave,mousemove");
                        });
                })
                .on("click", (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
        };

        bindHandler(hueInputContainer, hueIndicator, (newRatio) => {
            this.h = Math.round(newRatio * 360);
        });
        bindHandler(
            saturationInputContainer,
            saturationIndicator,
            (newRatio) => {
                this.s = newRatio;
            }
        );
        bindHandler(lightnessInputContainer, lightnessIndicator, (newRatio) => {
            this.l = newRatio;
        });

        return node.content(previewContainer);
    }
}

/**
 * This is the text input portion of the color, and shows the input as needed.
 */
class ColorInputView extends InputView {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-color-input";

    /**
     * @var class The class of the string input to use
     */
    static stringInputClass = StringInputView;

    /**
     * @var string The default value of the color, red
     */
    static defaultValue = "#ff0000";

    /**
     * When constructed, build both input classes.
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.stringInput = new this.constructor.stringInputClass(config, "hex", { value: this.value });
        let [h, s, l] = hexToHsl(this.value);

        this.colorInputHelper = new ColorInputHelperView(config, h, s, l);
        this.colorInputHelper.onChange((newValue) => {
            this.stringInput.setValue(newValue, false);
            this.value = newValue;
            this.changed();
        });

        this.stringInput.onFocus(async () => {
            if (!this.disabled) {
                let stringInputNode = await this.stringInput.getNode(),
                    colorInputHelperNode =
                        await this.colorInputHelper.getNode(),
                    inputPosition = this.node.element.getBoundingClientRect(),
                    left = inputPosition.x,
                    top = inputPosition.y + inputPosition.height,
                    width = inputPosition.width,
                    positionElement = (l, t, w) => {
                        left = l;
                        top = t;
                        width = w;
                        colorInputHelperNode.css({
                            width: `${w}px`,
                            left: `${l}px`,
                            top: `${t}px`
                        });
                    };

                this.repositionInterval = setInterval(() => {
                    inputPosition = this.node.element.getBoundingClientRect();
                    let thisLeft = inputPosition.x,
                        thisTop = inputPosition.y + inputPosition.height,
                        thisWidth = inputPosition.width;

                    if (
                        left !== thisLeft ||
                        top !== thisTop ||
                        width !== thisWidth
                    ) {
                        positionElement(thisLeft, thisTop, thisWidth);
                    }
                }, 25);

                document.body.appendChild(colorInputHelperNode.render());
                positionElement(left, top, width);
                colorInputHelperNode.css({
                    width: `${inputPosition.width}px`,
                    left: `${inputPosition.x}px`,
                    top: `${inputPosition.y + inputPosition.height}px`
                });

                let currentValue = strip(this.stringInput.getValue());
                if (!!currentValue.match(/^#[abcdefABCDEF0-9]{6}$/)) {
                    this.colorInputHelper.hex = currentValue;
                    this.value = currentValue;
                }

                this.colorInputHelper.show();

                let onClickElsewhereHideHelper = (e) => {
                    this.colorInputHelper.hide();
                    window.removeEventListener(
                        "click",
                        onClickElsewhereHideHelper,
                        false
                    );
                    clearInterval(this.repositionInterval);
                };

                window.addEventListener(
                    "click",
                    onClickElsewhereHideHelper,
                    false
                );
            }
        });

        this.stringInput.onInput(async () => {
            let currentValue = this.stringInput.getValue();
            if (!!currentValue.match(/^#[abcdefABCDEF0-9]{6}$/)) {
                this.colorInputHelper.hex = currentValue;
                this.value = currentValue;
                this.changed();
            }
        });

        this.stringInput.onBlur(async (e) => {
            let currentValue = strip(this.stringInput.getValue());
            if (!!currentValue.match(/^#[abcdefABCDEF0-9]{6}$/)) {
                this.colorInputHelper.hex = currentValue;
                this.value = currentValue;
                this.changed();
            } else {
                this.value = this.colorInputHelper.hex;
                this.stringInput.setValue(this.value, false);
                this.changed();
            }
            if (this.colorInputHelper.within) {
                this.colorInputHelper.hideOnLeave = true;
            } else {
                this.colorInputHelper.hide();
            }
        });
    }

    /**
     * Disable the sub-input when disable is called.
     */
    disable() {
        super.disable();
        this.stringInput.disable();
    }

    /**
     * Enable the sub-input when enable is called.
     */
    enable() {
        super.enable();
        this.stringInput.enable();
    }

    /**
     * Gets the value from memory
     */
    getValue() {
        return this.value;
    }

    /**
     * Override setValue to set the value to the appropriate nodes
     */
    setValue(value, triggerChange) {
        let result = super.setValue(value, false);
        if (this.stringInput !== undefined) {
            this.stringInput.setValue(value, false);
        }
        if (this.colorInputHelper !== undefined) {
            this.colorInputHelper.hex = value;
        }
        if (triggerChange) {
            this.changed();
        }
    }

    /**
     * When changed, if the color preview exists, change it.
     */
    changed() {
        if (this.node !== undefined) {
            let colorPreview = this.node.find(".inline-color-preview");
            if (colorPreview) {
                colorPreview.css("background-color", this.value);
            }
        }
        super.changed();
    }

    /**
     * Builds the node
     */
    async build() {
        let node = await super.build();
        node.append(
            E.span()
                .class("inline-color-preview")
                .css("background-color", this.value)
                .on("click", () => {
                    this.stringInput.focus();
                }),
            await this.stringInput.getNode()
        );

        return node;
    }
}

export { ColorInputView };
