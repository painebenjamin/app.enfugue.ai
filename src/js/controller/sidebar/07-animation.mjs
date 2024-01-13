/** @module controller/sidebar/04-animation */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import { VectorView } from "../../view/vector.mjs";
import { AnimationFormView } from "../../forms/enfugue/animation.mjs";

/**
 * Extends the menu controller for state and init
 */
class AnimationController extends Controller {
    /**
     * @var int extra space for the motion vector editor
     */
    static vectorPadding = 512;

    /**
     * Get data from the animation form
     */
    getState(includeImages = true) {
        return {
            "animation": this.animationForm.values,
            "motion": {
                "vector": this.vector.value
            }
        };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "motion": {
                "vector": []
            },
            "animation": {
                "animationEnabled": false,
                "animationEngine": "ad_hsxl",
                "animationFrames": 16,
                "animationRate": 8,
                "animationSlicing": true,
                "animationSize": 16,
                "animationStride": 8,
                "animationLoop": null,
                "animationRate": 8,
                "animationDecodeChunkSize": 1,
                "animationDenoisingIterations": 1,
                "animationInterpolation": null,
                "stableVideoAnimationFrames": 14,
                "stableVideoMotionBucketId": 127,
                "stableVideoFps": 7,
                "stableVideoNoiseAugStrength": 0.02,
                "stableVideoMinGuidanceScale": 1.0,
                "stableVideoMaxGuidanceScale": 3.0,
                "stableVideoUseDrag": false,
                "stableVideoReflect": false,
                "stableVideoModel": "svd",
                "stableVideoGaussianSigma": 20,
                "stableVideoRepeatVectors": true
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
        if (!isEmpty(newState.motion)) {
            this.vector.value = newState.motion.vector || [];
        }
    };

    /**
     * Resizes the vector view
     */
    resize(width=null, height=null) {
        this.vector.resizeCanvas(
            (width || this.engine.width) + this.constructor.vectorPadding * 2,
            (height || this.engine.height) + this.constructor.vectorPadding * 2
        );
    }

    /**
     * Gets vectors offset by padding
     */
    getOffsetVectors(vectors) {
        return vectors.map((points) => {
            return points.map((point) => {
                let offset = {
                    anchor: [
                        point.anchor[0]-this.constructor.vectorPadding,
                        point.anchor[1]-this.constructor.vectorPadding,
                    ]
                };
                if (!isEmpty(point.control_1)) {
                    offset.control_1 = [
                        point.control_1[0]-this.constructor.vectorPadding,
                        point.control_1[1]-this.constructor.vectorPadding,
                    ];
                }
                if (!isEmpty(point.control_2)) {
                    offset.control_2 = [
                        point.control_2[0]-this.constructor.vectorPadding,
                        point.control_2[1]-this.constructor.vectorPadding,
                    ];
                }
                return offset;
            });
        });
    }

    /**
     * Prepares a menu to be a motion vector menu
     */
    async prepareMenu(menu) {
        let lockVectors = await menu.addItem("Lock Motion Input", "fa-solid fa-lock"),
            clearVectors = await menu.addItem("Clear Motion Input", "fa-solid fa-delete-left");

        clearVectors.onClick(() => {
            this.vector.value = [];
        });
        lockVectors.onClick(() => {
            if (this.vector.hasClass("locked")) {
                this.vector.removeClass("locked");
                lockVectors.setIcon("fa-solid fa-lock");
            } else {
                this.vector.addClass("locked");
                lockVectors.setIcon("fa-solid fa-unlock");
            }
        });
    }

    /**
     * On init, append form and hide until SDXL gets selected
     */
    async initialize() {
        this.animationForm = new AnimationFormView(this.config);
        this.vector = new VectorView(
            this.config,
            this.engine.width + this.constructor.vectorPadding * 2,
            this.engine.height + this.constructor.vectorPadding * 2
        );
        this.vector.css({
            "left": `-${this.constructor.vectorPadding}px`,
            "top": `-${this.constructor.vectorPadding}px`
        });
        this.vector.hide();
        this.vector.onChange(() => this.animationForm.submit());
        this.application.sidebar.addChild(this.animationForm);
        this.animationForm.onSubmit(async (values) => {
            if (values.animationEnabled) {
                this.engine.animationFrames = values.animationFrames;
                this.engine.animationRate = values.animationRate;
                this.engine.animationInterpolation = values.animationInterpolation;
                this.engine.animationDecodeChunkSize = values.animationDecodeChunkSize;
                this.engine.animationInterpolation = values.animationInterpolation;

                if (isEmpty(values.animationEngine) || values.animationEngine === "ad_hsxl") {
                    this.vector.hide();
                    this.vectorToolbar.hide();
                    this.engine.animationEngine = "ad_hsxl";
                    this.engine.animationDenoisingIterations = values.animationDenoisingIterations;
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

                    if (values.animationSlicing || values.animationLoop) {
                        this.engine.animationSize = values.animationSize;
                        this.engine.animationStride = values.animationStride;
                    } else {
                        this.engine.animationSize = values.animationFrames;
                        this.engine.animationStride = 0;
                    }
               } else if (values.animationEngine === "svd") {
                    this.engine.fps = values.stableVideoFps;
                    this.engine.animationEngine = "svd";
                    this.engine.stableVideoModel = values.stableVideoModel;
                    this.engine.motionBucketId = values.stableVideoMotionBucketId;
                    this.engine.noiseAugStrength = values.stableVideoNoiseAugStrength;
                    this.engine.minGuidanceScale = values.stableVideoMinGuidanceScale;
                    this.engine.maxGuidanceScale = values.stableVideoMaxGuidanceScale;
                    if (values.stableVideoReflect) {
                        this.engine.animationLoop = "reflect";
                    } else {
                        this.engine.animationLoop = null;
                    }

                    if (values.stableVideoUseDrag) {
                        if (values.stableVideoRepeatVectors) {
                            let numCopies = Math.ceil(this.engine.animationFrames/14) - 1;
                            this.vector.setCopies(numCopies);
                            this.engine.motionVectors = this.getOffsetVectors(this.vector.extendedValue);
                        } else {
                            this.engine.motionVectors = this.getOffsetVectors(this.vector.value);
                        }
                        this.vector.show();
                        this.vectorToolbar.show();
                        this.application.container.classList.add("motion-vectors");
                        this.engine.gaussianSigma = values.stableVideoGaussianSigma;
                    } else {
                        this.application.container.classList.remove("motion-vectors");
                        this.vector.hide();
                        this.vectorToolbar.hide();
                        this.engine.motionVectors = null;
                    }
               }
            } else {
                this.application.container.classList.remove("motion-vectors");
                this.vector.hide();
                this.vectorToolbar.hide();
                this.engine.animationFrames = 0;
                this.engine.motionVectors = null;
            }
        });

        this.subscribe("engineAnimationEngineChange", (newEngine) => {
            setTimeout(() => {
                if (newEngine === "svd" && this.engine.animationFrames === 16) {
                    this.animationForm.setValues({...this.animationForm.values, ...{"animationFrames": 14}});
                } else if (newEngine === "ad_hsxl" && this.engine.animationFrames === 14) {
                    this.animationForm.setValues({...this.animationForm.values, ...{"animationFrames": 16}});
                }
            }, 125);
        });
        this.subscribe("engineWidthChange", (width) => { this.resize(width, null); });
        this.subscribe("engineHeightChange", (height) => { this.resize(null, height); });
        this.subscribe("engineAnimationFramesChange", (newFrames) => {
            if (this.animationForm.values.stableVideoRepeatVectors) {
                let numCopies = Math.ceil(newFrames/14) - 1;
                this.vector.setCopies(numCopies);
            }
        });
        this.subscribe("keyboard", (e) => {
            if (e.key == "c" && e.ctrlKey) {
                this.vector.copySelected();
            }
            if (e.key == "Delete") {
                this.vector.deleteSelected();
            }
        });

        this.vectorToolbar = new ToolbarView(this.config);
        this.vectorToolbar.addClass("motion-vectors");
        this.vectorToolbar.hide();

        await this.prepareMenu(this.vectorToolbar);

        this.application.container.appendChild(await this.vectorToolbar.render());
        (await this.canvas.getNode()).find("enfugue-image-editor-overlay").append(await this.vector.getNode());
    }
}

export { AnimationController as SidebarController };
