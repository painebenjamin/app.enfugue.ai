/** @module controlletr/sidebar/04-animation */
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
                "animationLoop": null
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
                this.engine.animationLoop = values.animationLoop;

                if (values.animationMotionScaleEnabled) {
                    this.engine.animationMotionScale = values.animationMotionScale;
                } else {

                    this.engine.animationMotionScale = null;
                }
                if (values.animationPositionEncodingSliceEnabled) {
                    this.engine.animationPositionEncodingTruncateLength = values.animationPositionEncodingTruncateLength;
                    this.engine.animationPositionEncodingScaleLength = values.animationPositionEncodingScaleLength;
                } else {
                    this.engine.animationPositionEncodingTruncateLength = null;
                    this.engine.animationPositionEncodingScaleLength = null;
                }

                if (values.animationChunking || values.animationLoop) {
                    this.engine.animationSize = values.animationSize;
                    this.engine.animationStride = values.animationStride;
                } else {
                    this.engine.animationSize = values.animationFrames;
                    this.engine.animationStride = null;
                }

                this.engine.animationInterpolation = values.animationInterpolation;
            } else {
                this.engine.animationFrames = 0;
            }
        });
        this.application.sidebar.addChild(this.animationForm);
    }
}

export { AnimationController as SidebarController };
