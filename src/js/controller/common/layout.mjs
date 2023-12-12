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
     * Sets the layout class
     */
    setLayout(newLayout) {
        if (this.layout !== newLayout) {
            this.application.container.classList.remove(`enfugue-layout-${this.layout}`);
            this.application.container.classList.add(`enfugue-layout-${newLayout}`);
            this.layout = newLayout;
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
        dynamic.onClick(() => this.setLayout("dynamic"));
        horizontal.onClick(() => this.setLayout("horizontal"));
        vertical.onClick(() => this.setLayout("vertical"));
        // Set defaults
        this.layout = "dynamic";
        this.horizontalRatio = 0.5;
        this.verticalRatio = 0.5;
        this.application.images.addClass("samples");
        this.application.canvas.hide();
    }
};

export { LayoutController };
