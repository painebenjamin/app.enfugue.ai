/** @module controller/common/animations */
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";

const E = new ElementBuilder();

/**
 * This class just manages CSS classes and enabling/disabling application animations
 */
class AnimationsController extends Controller {
    /**
     * Turns on/off animations
     */
    async toggleAnimations() {
        if (this.application.animations) {
            this.application.disableAnimations();
            this.animationStatusIndicator
                .removeClass("fa-play")
                .addClass("fa-stop")
                .data("tooltip", "Animations are disabled. Click to enable.");
        } else {
            this.application.enableAnimations();
            this.animationStatusIndicator
                .addClass("fa-play")
                .removeClass("fa-stop")
                .data("tooltip", "Animations are enabled. Click to disable.");
        }
    }

    /**
     * On initialization, append status indicator to header
     */
    async initialize() {
        this.animationStatusIndicator = E.i().class("fa-solid fa-play animations")
            .data("tooltip", "Animations are enabled. Click to disable.")
            .on("click", () => this.toggleAnimations());
        document.querySelector("header").appendChild(await this.animationStatusIndicator.render());
        if (this.application.session.getItem("animations") === false) {
            this.toggleAnimations();
        }
    }
}

export { AnimationsController };
