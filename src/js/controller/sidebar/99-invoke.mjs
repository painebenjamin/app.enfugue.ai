/** @module controller/sidebar/invoke */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { ButtonInputView } from "../../view/forms/input.mjs";
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
    getNodes() {
        let canvasState = this.images.getState();
        return canvasState.map((datum) => {
            let formattedState = {x: datum.x, y: datum.y, w: datum.w, h: datum.h};
            switch (datum.classname) {
                case "ImageEditorPromptNodeView":
                    formattedState["type"] = "prompt";
                    formattedState["infer"] = true;
                    formattedState["prompt"] = datum.prompt;
                    formattedState["negative_prompt"] = datum.negativePrompt;
                    formattedState["guidance_scale"] = datum.guidanceScale;
                    formattedState["inference_steps"] = datum.inferenceSteps;
                    formattedState["remove_background"] = datum.removeBackground;
                    break;
                case "ImageEditorScribbleNodeView":
                    formattedState["type"] = "scribble";
                    formattedState["image"] = datum.src;
                    formattedState["control"] = true;
                    formattedState["controlnet"] = "scribble";
                    formattedState["prompt"] = datum.prompt;
                    formattedState["negative_prompt"] = datum.negativePrompt;
                    formattedState["guidance_scale"] = datum.guidanceScale;
                    formattedState["inference_steps"] = datum.inferenceSteps;
                    formattedState["remove_background"] = datum.removeBackground;
                    break;
                case "ImageEditorImageNodeView":
                    formattedState["type"] = "image";
                    formattedState["fit"] = datum.fit;
                    formattedState["anchor"] = datum.anchor;
                    formattedState["infer"] = datum.infer;
                    formattedState["control"] = datum.control;
                    formattedState["inpaint"] = datum.inpaint;
                    formattedState["image"] = datum.src;
                    formattedState["mask"] = datum.scribbleSrc;
                    formattedState["strength"] = datum.strength;
                    formattedState["conditioning_scale"] = datum.conditioningScale;
                    formattedState["controlnet"] = datum.controlnet;
                    formattedState["prompt"] = datum.prompt;
                    formattedState["negative_prompt"] = datum.negativePrompt;
                    formattedState["guidance_scale"] = datum.guidanceScale;
                    formattedState["inference_steps"] = datum.inferenceSteps;
                    formattedState["remove_background"] = datum.removeBackground;
                    formattedState["process_control_image"] = datum.processControlImage;
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
        this.loadingBar.loading();
        this.invokeButton.disable().addClass("sliding-gradient");
        try {
            this.application.autosave();
            await this.engine.invoke({"nodes": this.getNodes()});
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
    }
}

export { InvokeButtonController as SidebarController }
