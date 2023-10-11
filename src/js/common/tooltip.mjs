/** @module common/tooltip */
import { SimpleNotification } from "./notify.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { isEmpty, stripHTML } from "../base/helpers.mjs";

const E = new ElementBuilder();

/**
 * This class enables any element to add 'data-tooltip' attributes, and a
 * tooltip will be shown near the cursor when it hovers over that element.
 */
class TooltipHelper {
    /**
     * @var int left offset in pixels
     */
    static offsetX = 10;

    /**
     * @var int top offset in pixels
     */
    static offsetY = 10;

    /**
     * @var int milliseconds to wait before showing tooltip
     */
    static activationDelay = 150;

    /**
     * @var int milliseconds to wait between intervals checking if the target node was removed
     */
    static deactivationDelay = 150;

    /**
     * @var string the tooltip container classname
     */
    static containerClassName = "tooltip-container";

    /**
     * On construct, bind mouse/touch events.
     */
    constructor() {
        this.tooltipContainer = E.div();
        this.tooltipContainer.class(this.constructor.containerClassName);

        window.addEventListener("mousemove", (e) => this.onMouseMove(e), true);
        window.addEventListener("mouseout", (e) => this.onMouseOut(e), true);
        window.addEventListener("touch", (e) => this.onTouch(e), true);
        window.addEventListener("contextmenu", (e) => this.onContextmenu(e), true);

        document.body.appendChild(this.tooltipContainer.render());
    }

    /**
     * On mouse out, check if the mouse has entirely left the window. If so, deactivate.
     */
    onMouseOut(e) {
        if (e.toElement === null) {
            this.deactivate();
        }
    }

    /**
     * On mouse move, perform check to see what tooltip to display, if any.
     */
    onMouseMove(e) {
        let tooltipTarget = e.target;
        while (!tooltipTarget.hasAttribute("data-tooltip")) {
            tooltipTarget = tooltipTarget.parentElement;
            if (isEmpty(tooltipTarget)) {
                break;
            }
        }
        if (isEmpty(tooltipTarget)) {
            this.deactivate();
            return;
        }
        this.setTooltip(tooltipTarget.getAttribute("data-tooltip"));
        this.x = e.clientX;
        this.y = e.clientY;
        this.positionElement();
    }

    /**
     * On touch, perform check to see what tooltip to display, if any.
     */
    onTouch(e) {
        let tooltipTarget = e.target;
        while (!tooltipTarget.hasAttribute("data-tooltip")) {
            tooltipTarget = tooltipTarget.parentElement;
            if (isEmpty(tooltipTarget)) {
                break;
            }
        }
        if (isEmpty(tooltipTarget)) {
            this.deactivate();
            return;
        }
        this.setTooltip(tooltipTarget.getAttribute("data-tooltip"));
        this.x = e.touches[0].clientX;
        this.y = e.touches[0].clientY;
        this.positionElement();
    }

    /**
     * On click, perform check to see if this is a copy.
     */
    onContextmenu(e) {
        if (!e.ctrlKey || !!!navigator.clipboard) return;
        let tooltipTarget = e.target;
        while (!tooltipTarget.hasAttribute("data-tooltip")) {
            tooltipTarget = tooltipTarget.parentElement;
            if (isEmpty(tooltipTarget)) {
                break;
            }
        }
        if (isEmpty(tooltipTarget)) {
            this.deactivate();
            return;
        }
        navigator.clipboard.writeText(stripHTML(tooltipTarget.getAttribute("data-tooltip")));
        SimpleNotification.notify("Copied to Clipboard", 1000);
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Sets the tooltip content and activates it.
     * The ElementBuilder checks, so we can call this all we need.
     */
    setTooltip(tooltipText) {
        if (tooltipText.indexOf("{") !== -1 && tooltipText.indexOf("}") !== -1) {
            // JSON, break all
            this.tooltipContainer.addClass("json");
        } else {
            this.tooltipContainer.removeClass("json");
        }
        if (!!navigator.clipboard) {
            tooltipText += "<br /><em class='note'>Ctrl+Right-Click to Copy Text</em>";
        }
        this.tooltipContainer.content(tooltipText);
        this.tooltipContainer.addClass("active");
    }

    /**
     * Position the element given an x and y poisiton.
     */
    positionElement() {
        if (this.x > window.innerWidth / 2) {
            this.tooltipContainer.css("left", null);
            this.tooltipContainer.css("right", window.innerWidth - this.x + this.constructor.offsetX);
        } else {
            this.tooltipContainer.css("right", null);
            this.tooltipContainer.css("left", this.x + this.constructor.offsetX);
        }

        if (this.y > window.innerHeight / 2) {
            this.tooltipContainer.css("top", null);
            this.tooltipContainer.css("bottom", window.innerHeight - this.y + this.constructor.offsetY);
        } else {
            this.tooltipContainer.css("bottom", null);
            this.tooltipContainer.css("top", this.y + this.constructor.offsetY);
        }
    }

    /**
     * Deactivate the tooltip by hiding the container.
     */
    deactivate() {
        this.tooltipContainer.removeClass("active");
    }
}

export { TooltipHelper };
