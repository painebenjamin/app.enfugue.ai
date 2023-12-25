/** @module controller/common/layout */
import { isEmpty, kebabCase, bindMouseUntilRelease } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { View } from "../../view/base.mjs";
import { ListInputView } from "../../forms/input.mjs";

const E = new ElementBuilder();
const leftMargin = 250;
const rightMargin = 250;
const topMargin = 96;
const bottomMargin = 80;

let currentHeight = window.innerHeight - (topMargin + bottomMargin),
    currentWidth = window.innerWidth - (leftMargin + rightMargin);

/**
 * Allows dragging the samples/canvas view
 */
class DragLayoutInputView extends View {
    static offsetLeft = -3;
    static offsetTop = -2;
    static minimumRatio = 0.1;

    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-drag-layout";

    /**
     * On construct, build buffers for callbacks
     */
    constructor(config) {
        super(config);
        this.onChangeCallbacks = [];
        this.mode = "horizontal";
        this.ratio = 0.5;
    }

    /**
     * Adds a change callback
     */
    onChange(callback) {
        this.onChangeCallbacks.push(callback);
    }

    /**
     * Sets the mode then resets rate
     */
    setMode(newMode, newRatio) {
        this.mode = newMode;
        this.setRatio(newRatio, false);
    }

    /**
     * Sets the current ratio
     */
    setRatio(newRatio, triggerCallbacks = true) {
        this.ratio = newRatio;
        if (triggerCallbacks) {
            for (let callback of this.onChangeCallbacks) {
                callback(this.ratio);
            }
        }
        if (!isEmpty(this.node)) {
            if (this.mode === "horizontal") {
                this.node.css({
                    "top": `${topMargin + 24}px`,
                    "bottom": `${bottomMargin}px`,
                    "right": "auto",
                    "left": `${leftMargin + (this.ratio * currentWidth)}px`
                });
            } else {
                this.node.css({
                    "top": `${topMargin + (this.ratio * currentHeight)}px`,
                    "bottom": "auto",
                    "left": `${leftMargin}px`,
                    "right": `${rightMargin}px`
                });
            }
        }
    }

    /**
     * 

    /**
     * On build, bind events
     */
    async build() {
        let node = await super.build();
        node.on("mousedown", (e) => {
            if (e.which !== 1) return;
            e.preventDefault();
            e.stopPropagation();
            bindMouseUntilRelease((e2) => {
                e2.preventDefault();
                e2.stopPropagation();
                let currentPosition = {
                        x: Math.max(e2.clientX - leftMargin, 0),
                        y: Math.max(e2.clientY - topMargin, 0)
                    },
                    currentRatio = {
                        x: Math.min(currentPosition.x, currentWidth) / currentWidth,
                        y: Math.min(currentPosition.y, currentHeight) / currentHeight
                    };

                currentRatio.x = Math.max(this.constructor.minimumRatio, Math.min(currentRatio.x, 1.0 - this.constructor.minimumRatio));
                currentRatio.y = Math.max(this.constructor.minimumRatio, Math.min(currentRatio.y, 1.0 - this.constructor.minimumRatio));

                if (this.mode === "horizontal") {
                    this.setRatio(currentRatio.x);
                } else {
                    this.setRatio(currentRatio.y);
                }
            });
        });
        return node;
    }
}

/**
 * Small input view for canvas/sample chooser
 */
class CurrentViewInputView extends ListInputView {
    /**
     * @var string CSS class
     */
    static className = "list-input-view current-view-input-view";

    /**
     * @var string Default is always canvas
     */
    static defaultValue = "canvas";

    /**
     * @var object Default options are just canvas
     */
    static defaultOptions = {
        "canvas": "Canvas"
    };

    /**
     * @var string The tooltip to display
     */
    static tooltip = "This is your current view. By default you will always view the input canvas. When there are results from your generations, the sample canvas will be available.";
}

/**
 * This class manages selecting between included and custom layouts
 */
class LayoutController extends Controller {
    /**
     * @var array All layouts
     */
    static layouts = ["dynamic", "vertical", "horizontal"];

    /**
     * Gets the current layout
     */
    get layout() {
        let storedLayout = this.application.session.getItem("layout");
        return isEmpty(storedLayout)
            ? "dynamic"
            : storedLayout;
    }

    /**
     * Sets the current layout
     */
    set layout(newLayout) {
        this.application.session.setItem("layout", newLayout);
        this.setLayout(newLayout);
    }

    /**
     * Gets the horizontal ratio
     */
    get horizontalRatio() {
        let storedHorizontalRatio = this.application.session.getItem("horizontalRatio");
        return isEmpty(storedHorizontalRatio)
            ? 0.5
            : storedHorizontalRatio;
    }

    /**
     * Sets the current horizontal ratio
     */
    set horizontalRatio(newHorizontalRatio) {
        this.application.session.setItem("horizontalRatio", newHorizontalRatio);
    }

    /**
     * Gets the vertical ratio
     */
    get verticalRatio() {
        let storedVerticalRatio = this.application.session.getItem("verticalRatio");
        return isEmpty(storedVerticalRatio)
            ? 0.5
            : storedVerticalRatio;
    }

    /**
     * Sets the current vertical ratio
     */
    set verticalRatio(newVerticalRatio) {
        this.application.session.setItem("verticalRatio", newVerticalRatio);
    }

    /**
     * Sets the layout class
     */
    setLayout(newLayout) {
        window.requestAnimationFrame(() => {
            for (let layout of this.constructor.layouts) {
                if (newLayout === layout) {
                    this.application.container.classList.add(`enfugue-layout-${layout}`);
                } else {
                    this.application.container.classList.remove(`enfugue-layout-${layout}`);
                }
            }
            if (newLayout === "dynamic") {
                this.hideSamples();
                this.dragLayoutView.hide();
            } else {
                this.showSamples();
                this.dragLayoutView.show();
                this.dragLayoutView.setMode(
                    newLayout,
                    newLayout === "horizontal"
                        ? this.horizontalRatio
                        : this.verticalRatio
                );
            }
            this.updateRatios();
        });
    }

    /**
     * Shows the sample canvas
     */
    showSamples(setCurrentView = true) {
        if (setCurrentView) {
            this.currentViewInput.setOptions({"canvas": "Canvas", "samples": "Samples"});
            this.currentViewInput.setValue("samples", false);
        }
        this.application.images.show();
    }

    /**
     * Hides the sample canvas
     */
    hideSamples(setCurrentView = true) {
        if (setCurrentView) {
            this.currentViewInput.setOptions({"canvas": "Canvas"});
            this.currentViewInput.setValue("canvas", false);
        }
        this.application.images.hide();
    }

    /**
     * Hides the sample canvas if necessary
     */
    checkHideSamples(setCurrentView = true) {
        if (this.layout === "dynamic") {
            this.hideSamples(setCurrentView);
        }
    }

    /**
     * Shows the sample canvas if necessary
     */
    checkShowSamples(setCurrentView = true) {
        if (this.layout !== "dynamic") {
            this.showSamples(setCurrentView);
        }
    }

    /**
     * Updates the display ratios
     */
    updateRatios(updateDragElement = false) {
        if (this.layout === "vertical") {
            this.canvas.node.css({
                "left": `${leftMargin}px`,
                "right": `${rightMargin}px`,
                "bottom": `${bottomMargin - 4 + ((1.0 - this.verticalRatio) * currentHeight)}px`,
                "top": `${topMargin}px`
            });
            this.images.node.css({
                "left": `${leftMargin}px`,
                "right": `${rightMargin}px`,
                "top": `${topMargin - 21 + (this.verticalRatio * currentHeight)}px`,
                "bottom": `${bottomMargin}px`
            });
            if (updateDragElement) {
                this.dragLayoutView.setRatio(this.verticalRatio, false);
            }
        } else if (this.layout === "horizontal") {
            this.canvas.node.css({
                "left": `${leftMargin}px`,
                "bottom": `${bottomMargin}px`,
                "top": `${topMargin}px`,
                "right": `${rightMargin - 2 + ((1.0 - this.horizontalRatio) * currentWidth)}px`,
            });
            this.images.node.css({
                "left": `${leftMargin + 2 + (this.horizontalRatio * currentWidth)}px`,
                "right": `${rightMargin}px`,
                "top": `${topMargin}px`,
                "bottom": `${bottomMargin}px`
            });
            if (updateDragElement) {
                this.dragLayoutView.setRatio(this.horizontalRatio, false);
            }
        } else {
            this.canvas.node.css({
                "left": null,
                "right": null,
                "top": null,
                "bottom": null
            });
            this.images.node.css({
                "left": null,
                "right": null,
                "top": null,
                "bottom": null
            });
        }
    }

    /**
     * On initialization, add menu items and bind events
     */
    async initialize() {
        let layoutMenu = this.application.menu.getCategory("Layout"),
            dynamic = await layoutMenu.addItem("Dynamic", "fa-solid fa-arrows-rotate"),
            horizontal = await layoutMenu.addItem("Split Horizontally", "fa-solid fa-arrows-left-right"),
            vertical = await layoutMenu.addItem("Split Vertically", "fa-solid fa-arrows-up-down"),
            tileHorizontal = await layoutMenu.addItem("Tile Horizontally", "fa-solid fa-ellipsis"),
            tileVertical = await layoutMenu.addItem("Tile Vertically", "fa-solid fa-ellipsis-vertical");

        dynamic.onClick(() => { this.layout = "dynamic"; });
        horizontal.onClick(() => { this.layout = "horizontal"; });
        vertical.onClick(() => { this.layout = "vertical"; });
        tileHorizontal.onClick(() => {
            let isActive = tileHorizontal.hasClass("active");
            this.application.samples.setTileHorizontal(!isActive);
            tileHorizontal.toggleClass("active");
        });
        tileVertical.onClick(() => {
            let isActive = tileVertical.hasClass("active");
            this.application.samples.setTileVertical(!isActive);
            tileVertical.toggleClass("active");
        });

        // Add view selector
        this.currentViewInput = new CurrentViewInputView(this.config);
        this.currentViewInput.onChange((_, value) => {
            if (value === "samples") {
                this.showSamples(false);
            } else {
                this.application.samples.setPlay(false);
                setTimeout(() => {
                    this.hideSamples(false);
                }, 100);
            }
        });
        this.application.container.appendChild(await this.currentViewInput.render());

        // Add layout dragger
        this.dragLayoutView = new DragLayoutInputView(this.config);
        this.dragLayoutView.onChange((newRatio) => {
            if (this.layout === "horizontal") {
                this.horizontalRatio = newRatio;
            } else {
                this.verticalRatio = newRatio;
            }
            this.updateRatios();
        });
        this.dragLayoutView.hide();
        this.application.container.appendChild(await this.dragLayoutView.render());
        this.setLayout(this.layout);
        this.application.images.addClass("samples");
        this.checkHideSamples();

        window.addEventListener("resize", () => {
            currentHeight = window.innerHeight - (topMargin + bottomMargin);
            currentWidth = window.innerWidth - (leftMargin + rightMargin);
            this.updateRatios(true);
        });
    }
};

export { LayoutController };
