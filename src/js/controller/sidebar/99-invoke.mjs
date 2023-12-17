/** @module controller/sidebar/99-invoke */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { ButtonInputView } from "../../forms/input.mjs";
import { Controller } from "../base.mjs";
import { View } from "../../view/base.mjs";

const E = new ElementBuilder();

/**
 * This class adds a view just for the loading bar to ensure the user knows it's loading
 */
class InvokeLoadingBarView extends View {
    /**
     * @var string custom class name
     */
    static className = "invoke-loader";

    /**
     * @var string custom class name when calling .loading()
     */
    static loaderClassName = "loading-bar";
}

/**
 * This is just the button itself
 */
class EnfugueButton extends ButtonInputView {
    /**
     * @var string The class name of the view.
     */
    static className = "invoke";

    /**
     * @var string the value passed to the 'value' property, in this case the text.
     */
    static defaultValue = "ENFUGUE";
}

/**
 * The Invoke controller reads the canvas and sends the data to the invocation engine.
 */
class InvokeButtonController extends Controller {
    /**
     * Gets the step data from the canvas for invocation.
     */
    async getLayers() {
        let layerState = this.application.layers.getState(),
            unusedLayers = [],
            mapped = layerState.layers.map((datum, i) => {
                let formattedState = {
                    "x": datum.x,
                    "y": datum.y,
                    "w": datum.w,
                    "h": datum.h,
                    "remove_background": datum.removeBackground,
                    "image": datum.src
                };
                
                switch (datum.classname) {
                    case "ImageEditorScribbleNodeView":
                        formattedState["control_units"] = [
                            {"process": false, "controlnet": "scribble"}
                        ];
                        break;
                    case "ImageEditorImageNodeView":
                    case "ImageEditorVideoNodeView":
                        formattedState["fit"] = datum.fit;
                        formattedState["anchor"] = datum.anchor;
                        formattedState["opacity"] = datum.opacity;
                        formattedState["visibility"] = datum.visibility;
                        formattedState["frame"] = isEmpty(datum.startFrame) ? 0 : datum.startFrame - 1;

                        if (datum.imagePrompt) {
                            formattedState["ip_adapter_scale"] = datum.imagePromptScale;
                        }
                        if (datum.control) {
                            formattedState["control_units"] = datum.controlnetUnits.map((unit) => {
                                return {
                                    "process": unit.processControlImage,
                                    "start": unit.conditioningStart,
                                    "end": unit.conditioningEnd,
                                    "scale": unit.conditioningScale,
                                    "controlnet": unit.controlnet
                                };
                            });
                        }
                        if (!isEmpty(datum.skipFrames)) {
                            formattedState["skip_frames"] = datum.skipFrames;
                        }
                        if (!isEmpty(datum.divideFrames)) {
                            formattedState["divide_frames"] = datum.divideFrames;
                        }
                        if (
                            ["visible", "denoised"].indexOf(datum.visibility) === -1 &&
                            isEmpty(formattedState.ip_adapter_scale) &&
                            isEmpty(formattedState.control_units)
                        ) {
                            unusedLayers.push(i);
                        }
                        break;
                    default:
                        throw `Unknown classname ${datum.classname}`;
                }
                return formattedState;
            });

        if (!isEmpty(unusedLayers)) {
            let s = unusedLayers.length === 1 ? "" : "s",
                s_ve = unusedLayers.length === 1 ? "s" : "ve",
                this_these = unusedLayers.length === 1 ? "this" : "these",
                was_were = unusedLayers.length === 1 ? "was" : "were",
                it_them = unusedLayers.length === 1 ? "it" : "them";

            if (!(await this.confirm(
                `${unusedLayers.length} layer${s} ha${s_ve} no role assigned, ` +
                `${this_these} layer${s} will not be not sent to the backend. ` +
                `Add a role to ${this_these} layer${s} to use ${it_them}, like selecting ` +
                `"Visible" or "Denoised" visibility mode, using it with IP ` +
                `Adapter, and/or assigning one or more control units.` +
                `<br /><br />Continue anyway?`
            ))) {
                throw "Invocation canceled.";
            }
            return mapped.filter((v, i) => unusedLayers.indexOf(i) === -1);
        }
        return mapped;
    }
    
    /**
     * Tries to invoke the engine.
     * Can restart itself depending on errors
     */
    async tryInvoke() {
        this.isInvoking = true;
        this.loadingBar.loading();
        this.invokeButton.disable().addClass("sliding-gradient");
        try {
            this.application.autosave();
            await this.application.invoke({"layers": await this.getLayers()});
        } catch(e) {
            console.error(e);
            let errorMessage = `${e}`;
            if (!isEmpty(e.detail)) {
                errorMessage = e.detail;
            } else if(!isEmpty(e.title)) {
                errorMessage = e.title;
            }
            if (errorMessage.toLowerCase().indexOf("engine process died") !== -1) {
                // Try again
                this.notify("warn", "Engine Didn't Start", "The diffusion engine process exited before it started responding to requests. Waiting a moment and trying again.");
                return await this.tryInvoke();
            } else {
                this.notify("error", "Couldn't Start", errorMessage);
            }
        }
        this.invokeButton.enable().removeClass("sliding-gradient");
        this.loadingBar.doneLoading();
        this.application.autosave();
        this.isInvoking = false;
    }

    /**
     * On initialize, build button and bind actions.
     */
    async initialize() {
        this.invokeButton = new EnfugueButton(this.config);
        this.invokeButton.onChange(() => this.tryInvoke());
        this.loadingBar = new InvokeLoadingBarView();
        await this.application.sidebar.addChild(this.invokeButton);
        await this.application.sidebar.addChild(this.loadingBar);
        this.subscribe("tryInvoke", () => {
            if (this.isInvoking !== true) {
                this.tryInvoke();
            }
        });
    }
}

export { InvokeButtonController as SidebarController };
