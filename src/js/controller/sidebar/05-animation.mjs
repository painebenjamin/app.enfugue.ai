/** @module controlletr/sidebar/05-animation */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { AnimationFormView } from "../../forms/enfugue/animation.mjs";

/**
 * Extends the menu controller for state and init
 */
class AnimationController extends Controller {
    /**
     * Get data from the animation form
     */
    getState(includeImages = true) {
        return { "animation": this.animationForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "animation": {
                "animationEnabled": false,
                "animationFrames": 16,
                "animationRate": 8,
                "animationChunking": true,
                "animationSize": 16,
                "animationStride": 8,
                "animationLoop": false
            }
        };
    }

    /**
     * Set state in the animation form
     */
    setState(newState) {
        if (!isEmpty(newState.animation)) {
            this.animationForm.setValues(newState.animation).then(() => this.animationForm.submit());
        }
    };

    /**
     * On init, append form and hide until SDXL gets selected
     */
    async initialize() {
        this.animationForm = new AnimationFormView(this.config);
        this.animationForm.onSubmit(async (values) => {
            if (values.animationEnabled) {
                this.engine.animationFrames = values.animationFrames;
                this.engine.animationRate = values.animationRate;
                if (values.animationChunking) {
                    this.engine.animationSize = values.animationSize;
                    this.engine.animationStride = values.animationStride;
                    this.engine.animationLoop = values.animationLoop;
                } else {
                    this.engine.animationSize = null;
                    this.engine.animationLoop = false;
                }
            } else {
                this.engine.animationFrames = 0;
            }
        });
        this.application.sidebar.addChild(this.animationForm);
    }
}

export { AnimationController as SidebarController };
