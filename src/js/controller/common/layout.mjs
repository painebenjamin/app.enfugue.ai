/** @module controller/common/layout */
import { isEmpty, kebabCase } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";

const E = new ElementBuilder();

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
        });
    }

    /**
     * Shows the sample canvas if necessary
     */
    checkShowSamples() {
        if (this.layout === "dynamic") {
            this.application.canvas.hide();
        } else {
            this.application.canvas.show();
        }
    }

    /**
     * On initialization, add menu items and bind events
     */
    async initialize() {
        let layoutMenu = this.application.menu.getCategory("Layout"),
            dynamic = await layoutMenu.addItem("Dynamic", "fa-solid fa-arrows-rotate"),
            horizontal = await layoutMenu.addItem("Split Horizontally", "fa-solid fa-arrows-left-right"),
            vertical = await layoutMenu.addItem("Split Vertically", "fa-solid fa-arrows-up-down");

        dynamic.onClick(() => { this.layout = "dynamic"; });
        horizontal.onClick(() => { this.layout = "horizontal"; });
        vertical.onClick(() => { this.layout = "vertical"; });

        // Set defaults
        this.horizontalRatio = 0.5;
        this.verticalRatio = 0.5;
        this.setLayout(this.layout);
        this.application.images.addClass("samples");
        this.checkShowSamples();
    }
};

export { LayoutController };
