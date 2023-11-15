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
    getLayers() {
        let layerState = this.application.layers.getState();
        console.log(layerState);
        return layerState.layers.map((datum, i) => {
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
                    formattedState["denoise"] = !!datum.denoise;
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
                    break;
                default:
                    throw `Unknown classname ${datum.classname}`;
            }
            return formattedState;
        });
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
            await this.application.invoke({"layers": this.getLayers()});
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
