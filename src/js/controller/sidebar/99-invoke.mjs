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
    getNodes() {
        let nodes = [];
        return nodes.map((datum, i) => {
            let formattedState = {
                "x": datum.x,
                "y": datum.y,
                "w": datum.w,
                "h": datum.h,
                "inference_steps": datum.inferenceSteps,
                "guidance_scale": datum.guidanceScale,
                "scale_to_model_size": datum.scaleToModelSize,
                "remove_background": datum.removeBackground
            };

            if (Array.isArray(datum.prompt)) {
                formattedState["prompt"], formattedState["prompt_2"] = datum.prompt;
            } else {
                formattedState["prompt"] = datum.prompt;
            }
           
            if (Array.isArray(datum.negativePrompt)) {
                formattedState["negative_prompt"], formattedState["negative_prompt_2"] = datum.negativePrompt;
            } else {
                formattedState["negative_prompt"] = datum.negativePrompt;
            }
            
            switch (datum.classname) {
                case "ImageEditorPromptNodeView":
                    break;
                case "ImageEditorScribbleNodeView":
                    formattedState["control_images"] = [
                        {"image": datum.src, "process": false, "invert": true, "controlnet": "scribble"}
                    ];
                    break;
                case "ImageEditorImageNodeView":
                    formattedState["fit"] = datum.fit;
                    formattedState["anchor"] = datum.anchor;
                    if (datum.infer || datum.inpaint || (!datum.infer && !datum.inpaint && !datum.imagePrompt && !datum.control)) {
                        formattedState["image"] = datum.src;
                    }
                    if (datum.infer) {
                        formattedState["strength"] = datum.strength;
                    }
                    if (datum.inpaint) {
                        formattedState["mask"] = datum.scribbleSrc;
                        formattedState["invert_mask"] = true; // The UI is inversed
                        formattedState["crop_inpaint"] = datum.cropInpaint;
                        formattedState["inpaint_feather"] = datum.inpaintFeather;
                    }
                    if (datum.imagePromptPlus) {
                        formattedState["ip_adapter_plus"] = true;
                        if (datum.imagePromptFace) {
                            formattedState["ip_adapter_face"] = true;
                        }
                    }
                    if (datum.imagePrompt) {
                        formattedState["ip_adapter_images"] = [
                            {
                                "image": datum.src,
                                "scale": datum.imagePromptScale,
                                "fit": datum.fit,
                                "anchor": datum.anchor
                            }
                        ];
                    }
                    if (datum.control) {
                        formattedState["control_images"] = [
                            {
                                "image": datum.src,
                                "process": datum.processControlImage,
                                "invert": datum.invertControlImage === true,
                                "controlnet": datum.controlnet,
                                "scale": datum.conditioningScale,
                                "fit": datum.fit,
                                "anchor": datum.anchor,
                                "start": datum.conditioningStart,
                                "end": datum.conditioningEnd
                            }
                        ];
                    }
                    break;
                case "ImageEditorCompoundImageNodeView":
                    let imageNodeIndex, promptImageNodeIndex;
                    for (let j = 0; j < datum.children.length; j++) {
                        let child = datum.children[j];
                        if (child.infer || child.inpaint) {
                            if (!isEmpty(imageNodeIndex)) {
                                messages.push(`Node {i+1}: Base image set in image {imageNodeIndex+1}, ignoring additional set in {j+1}`);
                            } else {
                                imageNodeIndex = j;
                                formattedState["image"] = child.src;
                                formattedState["anchor"] = child.anchor;
                                formattedState["fit"] = child.fit;
                            }
                        }
                        if (child.infer && imageNodeIndex == j) {
                            formattedState["strength"] = child.strength;
                        }
                        if (child.inpaint && imageNodeIndex == j) {
                            formattedState["mask"] = child.scribbleSrc;
                            formattedState["invert_mask"] = true; // The UI is inversed
                            formattedState["crop_inpaint"] = child.cropInpaint;
                            formattedState["inpaint_feather"] = child.inpaintFeather;
                        }
                        if (child.imagePrompt) {
                            if (isEmpty(formattedState["ip_adapter_images"])) {
                                formattedState["ip_adapter_images"] = [];
                            }
                            if (child.imagePromptPlus) {
                                formattedState["ip_adapter_plus"] = true;
                                if (child.imagePromptFace) {
                                    formattedState["ip_adapter_face"] = true;
                                }
                            }
                            formattedState["ip_adapter_images"].push(
                                {
                                    "image": child.src,
                                    "scale": child.imagePromptScale,
                                    "fit": child.fit,
                                    "anchor": child.anchor
                                }
                            );
                        }
                        if (child.control) {
                            if (isEmpty(formattedState["control_images"])) {
                                formattedState["control_images"] = [];
                            }
                            formattedState["control_images"].push(
                                {
                                    "image": child.src,
                                    "process": child.processControlImage,
                                    "invert": child.colorSpace == "invert",
                                    "controlnet": child.controlnet,
                                    "scale": child.conditioningScale,
                                    "fit": child.fit,
                                    "anchor": child.anchor
                                }
                            );
                        }
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
            await this.application.invoke({"nodes": this.getNodes()});
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
