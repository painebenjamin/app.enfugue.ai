/** @module controller/sidebar/05-tweaks */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import {
    TweaksFormView,
    SchedulerConfigurationFormView
} from "../../forms/enfugue/tweaks.mjs";

const E = new ElementBuilder();

/**
 * Adds an advanced scheduler button
 */
class AdvancedTweaksFormView extends TweaksFormView {
    /**
     * @var int Width of the scheduler window
     */
    static schedulerWindowWidth = 400;

    /**
     * @var int Height of the scheduler window
     */
    static schedulerWindowHeight = 140;

    /**
     * Gets the scheduler form
     */
    getSchedulerForm() {
        if (isEmpty(this.schedulerForm)) {
            this.schedulerForm = new SchedulerConfigurationFormView(this.config);
            this.schedulerForm.onSubmit((values) => {
                this.values = {...this.values,...values};
                this.submit();
            });
        }
        return this.schedulerForm; 
    }

    /**
     * Sets values
     */
    async setValues(newValues, trigger) {
        this.getSchedulerForm().setValues(newValues, false);
        return await super.setValues(newValues, trigger);
    }

    /**
     * Shows the more scheduler configuration form
     */
    async showSchedulerConfiguration() {
        if (!isEmpty(this.schedulerWindow)) {
            this.schedulerWindow.focus();
        } else {
            let schedulerForm = this.getSchedulerForm();
            schedulerForm.setValues(this.values, false);
            this.schedulerWindow = await this.spawnWindow(
                "More Scheduler Configuration",
                schedulerForm,
                this.constructor.schedulerWindowWidth,
                this.constructor.schedulerWindowHeight
            );
            this.schedulerWindow.onClose(() => {
                delete this.schedulerWindow;
            });
        }
    }

    /**
     * On build, append button
     */
    async build() {
        let node = await super.build(),
            moreSchedulerConfig = E.i().class("fa-solid fa-gear").css({
                "position": "absolute",
                "cursor": "pointer",
                "top": 0,
                "right": 0,
            })
            .data("tooltip", "More Scheduler Configuration")
            .on("click", (e) => {
                this.showSchedulerConfiguration();
            });

        node.find(".scheduler-input-view").append(moreSchedulerConfig);
        return node;
    }
}

/**
 * Extend the menu controll to bind init
 */
class TweaksController extends Controller {
    /**
     * Return data from the tweaks form
     */
    getState(includeImages = true) {
        return { "tweaks": this.tweaksForm.values };
    }

    /**
     * Sets state in the form
     */
    setState(newState) {
        if (!isEmpty(newState.tweaks)) {
            this.tweaksForm.setValues(newState.tweaks).then(() => this.tweaksForm.submit());
        }
    }

    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "tweaks": {
                "guidanceScale": this.config.model.invocation.guidanceScale,
                "inferenceSteps": this.config.model.invocation.inferenceSteps,
                "scheduler": null,
                "clipSkip": 0,
                "enableFreeU": false,
                "freeUBackbone1": 1.5,
                "freeUBackbone2": 1.6,
                "freeUSkip1": 0.9,
                "freeUSkip2": 0.2,
                "noiseOffset": 0.0,
                "noiseMethod": "simplex",
                "noiseBlendMethod": "inject",
                "betaStart": null,
                "betaEnd": null,
                "betaSchedule": null
            }
        }
    }

    /**
     * On initialization, append the Tweaks form
     */
    async initialize() {
        // Builds form
        this.tweaksForm = new AdvancedTweaksFormView(this.config);
        this.tweaksForm.spawnWindow = (name, content, w, h, x, y) => {
            return this.spawnWindow(name, content, w, h, x, y);
        }
        this.tweaksForm.onSubmit(async (values) => {
            this.engine.guidanceScale = values.guidanceScale;
            this.engine.inferenceSteps = values.inferenceSteps;
            this.engine.scheduler = values.scheduler;
            this.engine.clipSkip = values.clipSkip;
            this.engine.noiseOffset = values.noiseOffset;
            this.engine.noiseMethod = values.noiseMethod;
            this.engine.noiseBlendMethod = values.noiseBlendMethod;

            this.engine.betaStart = values.betaStart;
            this.engine.betaEnd = values.betaEnd;
            this.engine.betaSchedule = values.betaSchedule;

            if (values.enableFreeU) {
                this.engine.freeUFactors = [
                    values.freeUBackbone1,
                    values.freeUBackbone2,
                    values.freeUSkip1,
                    values.freeUSkip2
                ];
            } else {
                this.engine.freeUFactors = null;
            }
        });

        // Subscribe to model changes to look for defaults
        this.subscribe("modelPickerChange", (newModel) => {
            if (!isEmpty(newModel)) {
                let defaultConfig = newModel.defaultConfiguration,
                    tweaksConfig = {};
                
                if (!isEmpty(defaultConfig.guidance_scale)) {
                    tweaksConfig.guidanceScale = defaultConfig.guidance_scale;
                }
                if (!isEmpty(defaultConfig.num_inference_steps)) {
                    tweaksConfig.inferenceSteps = defaultConfig.num_inference_steps;
                }
                if (!isEmpty(defaultConfig.clip_skip)) {
                    tweaksConfig.clipSkip = defaultConfig.clip_skip;
                }
                if (!isEmpty(defaultConfig.noise_offset)) {
                    tweaksConfig.noiseOffset = defaultConfig.noise_offset;
                }
                if (isEmpty(defaultConfig.noise_method)) {
                    tweaksConfig.noiseMethod = this.tweaksForm.values.noiseMethod;
                } else {
                    tweaksConfig.noiseMethod = defaultConfig.noise_method;
                }
                if (isEmpty(defaultConfig.noise_blend_method)) {
                    tweaksConfig.noiseBlendMethod = this.tweaksForm.values.noiseBlendMethod;
                } else {
                    tweaksConfig.noiseBlendMethod = defaultConfig.noise_blend_method;
                }
                if (!isEmpty(defaultConfig.freeu_factors)) {
                    tweaksConfig.enableFreeU = true;
                    tweaksConfig.freeUBackbone1 = defaultConfig.freeu_factors[0];
                    tweaksConfig.freeUBackbone2 = defaultConfig.freeu_factors[1];
                    tweaksConfig.freeUSkip1 = defaultConfig.freeu_factors[2];
                    tweaksConfig.freeUSkip2 = defaultConfig.freeu_factors[3];
                } else {
                    tweaksConfig.enableFreeU = false;
                }
                if (isEmpty(newModel.scheduler)) {
                    tweaksConfig.scheduler = this.tweaksForm.values.scheduler;
                } else {
                    tweaksConfig.scheduler = newModel.scheduler[0].name;
                }

                if (!isEmpty(tweaksConfig)) {
                    this.tweaksForm.setValues(tweaksConfig);
                }
            }
        });

        // Add to sidebar
        this.application.sidebar.addChild(this.tweaksForm);
    }
}

export { TweaksController as SidebarController }
