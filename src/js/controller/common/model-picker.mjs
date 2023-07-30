/** @module controller/common/module-picker */
import { Controller } from "../base.mjs";
import { TableView } from "../../view/table.mjs";
import { View } from "../../view/base.mjs";
import { FormView } from "../../view/forms/base.mjs";
import {
    SearchListInputView, 
    StringInputView, 
    SearchListInputListView,
    NumberInputView
} from "../../view/forms/input.mjs";
import { 
    VAEInputView,
    CheckpointInputView,
    MultiLoraInputView,
    MultiLycorisInputView,
    MultiInversionInputView
} from "./model-manager.mjs";
import { isEmpty, waitFor, createElementsFromString } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";

const E = new ElementBuilder();

/**
 * Extend the SearchListInputListView to add additional classes
 */
class ModelPickerListInputView extends SearchListInputListView {
    /**
     * @var array<string> CSS classes
     */
    static classList = SearchListInputListView.classList.concat(["model-picker-list-input-view"]);
};

/**
 * Extend the StringInputView so we can strip HTML from the value
 */
class ModelPickerStringInputView extends StringInputView {
    /**
     * Strip HTML from the value and only display the name portion.
     */
    setValue(newValue, triggerChange) {
        if(!isEmpty(newValue)) {
            if (newValue.startsWith("<")) {
                newValue = createElementsFromString(newValue)[0].innerText;
            } else {
                newValue = newValue.split("/")[1];
            }
        }
        return super.setValue(newValue, triggerChange);
    }
};

/**
 * We extend the SearchListInputView to change some default config.
 */
class ModelPickerInputView extends SearchListInputView {
    /**
     * @var string The content of the node when nothing is selected.
     */
    static placeholder = "Start typing to search models…";

    /**
     * @var class The class of the string input, override so we can override setValue
     */
    static stringInputClass = ModelPickerStringInputView;

    /**
     * @var class The class of the list input, override so we can add css classes
     */
    static listInputClass = ModelPickerListInputView
};

/**
 * This form allows additional pipeline configuration when using a checkpoint
 */
class ModelConfigurationFormView extends FormView {
    /**
     * @var string Custom CSS class
     */
    static className = "model-configuration-form-view";

    /**
     * @var boolean no submit button
     */
    static autoSubmit = true;

    /**
     * @var boolean Start hidden
     */
    static collapseFieldSets = true;

    /**
     * @var object one fieldset describes all inputs
     */
    static fieldSets = {
        "Adaptations and Modifications": {
            "lora": {
                "class": MultiLoraInputView,
                "label": "LoRA",
                "config": {
                    "tooltip": "LoRA stands for <strong>Low Rank Adapation</strong>, it is a kind of fine-tuning that can perform very specific modifications to Stable Diffusion such as training an individual's appearance, new products that are not in Stable Diffusion's training set, etc."
                }
            },
            "lycoris": {
                "class": MultiLycorisInputView,
                "label": "LyCORIS",
                "config": {
                    "tooltip": "LyCORIS stands for <strong>LoRA beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion</strong>, a novel means of performing low-rank adaptation introduced in early 2023."
                }
            },
            "inversion": {
                "class": MultiInversionInputView,
                "label": "Textual Inversion",
                "config": {
                    "tooltip": "Textual Inversion is another kind of fine-tuning that teaches novel concepts to Stable Diffusion in a small number of images, which can be used to positively or negatively affect the impact of various prompts."
                }
            }
        },
        "Additional Models": {
            "vae": {
                "label": "VAE",
                "class": VAEInputView
            },
            "refiner": {
                "label": "Refining Checkpoint",
                "class": CheckpointInputView,
                "config": {
                    "tooltip": "Refining checkpoints were introduced with SDXL 0.9 - these are checkpoints specifically trained to improve detail, shapes, and generally improve the quality of images generated from the base model. These are optional, and do not need to be specifically-trained refinement checkpoints - you can try mixing and matching checkpoints for different styles, though you may wish to ensure the related checkpoints were trained on the same size images."
                }
            },
            "inpainter": {
                "label": "Inpainting Checkpoint",
                "class": CheckpointInputView,
                "config": {
                    "tooltip": "An inpainting checkpoint if much like a regular Stable Diffusion checkpoint, but it additionally includes the ability to input which parts of the image can be changed and which cannot. This is used when you specifically request an image be inpainted, but is also used in many other situations in Enfugue; such as when you place an image on the canvas that doesn't cover the entire space, or use an image that has transparency in it (either before or after removing it's background.) When you don't select an inpainting checkpoint and request an inpainting operation, one will be created dynamically from the main checkpoint at runtime."
                }
            }
        }
    };
};

/**
 * Extend the TableView to disable sorting and add conditional buttons
 */
class ModelTensorRTTableView extends TableView {
    /**
     * Add a parameter for the engine build callable
     */
    constructor(config, data, buildEngine) {
        super(config, data);
        this.buildEngine = buildEngine;
    }

    /**
     * @var bool Disable sorting.
     */
    static canSort = false;

    /**
     * @var object Column formatter for build buttons
     */
    static columnFormatters = {
        "Build": function(value, row) {
            if (value === true) {
                return E.span().content("Ready");
            } else {
                let button = E.button().content("Build").on("click", async () => {
                    try {
                        button.disabled(true).addClass("loading-bar loading");
                        await this.buildEngine(row);
                        button.removeClass("loading-bar").removeClass("loading").content("Ready");
                    } catch(e) {
                        button.removeClass("loading-bar loading").disabled(false);
                    }
                });
                if (value === "building") {
                    button.disabled(true).addClass("loading-bar loading");
                }
                return button;
            }
        }
    };
};

/**
 * The Status View shows each engine and their status.
 */
class ModelTensorRTStatusView extends View {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-tensorrt-status-view";

    /**
     * @var object The supported network names
     */
    static supportedNetworks = {
        "unet": "UNet",
        "controlled_unet": "Controlled UNet",
        "inpaint_unet": "Inpainting UNet"
    };

    /**
     * @var string The base description showed above the table
     */
    static tensorRTDescription = [
        "TensorRT is a technology created by Nvidia that transforms an AI model into one that takes advantage of hardware acceleration available on Nvidia GPUs.",
        "As there are numerous varying architectures used by Nvidia that support this technology, these engines must be compiled by an architecture compatible with your actual hardware, rather than distributed by AI model providers.",
        "The compilation time for each model varies, but generally takes between 15 and 30 minutes each. You can expect between 50% and 100% faster inference speeds during the engine's respective step(s).",
        "The compiled engine is only useable for a model with the same checkpoint, LoRA, LyCORIS, Textual Inversion and engine size. If you change any of those details about this model, it will require recompilation. You can safely change model prompts as desired without requiring a new engine.",
    ];

    /**
     * @var object The text descriptions of the supported networks
     */
    static networkDescriptions = {
        "unet": "The network used when creating images with a prompt or base image.",
        "controlled_unet": "The network used when creating images with a control image.",
        "inpaint_unet": "The network used when inpainting or outpainting."
    };

    /**
     * Constructed after acquiring status.
     */
    constructor(config, status, buildEngine) {
        super(config);
        this.status = status;
        this.buildEngine = buildEngine;
    }

    /**
     * Gets the table data for the table view
     */
    get tableData() {
        return Object.getOwnPropertyNames(this.constructor.supportedNetworks).map((networkName) => {
            return {
                "Network Name": this.constructor.supportedNetworks[networkName],
                "Description": this.constructor.networkDescriptions[networkName],
                "Build": this.status["building"] === networkName
                    ? "building"
                    : this.status[`${networkName}_ready`]
            };
        });
    }

    /**
     * Gets the network key from the label
     */
    getNameFromLabel(label) {
        for (let supportedNetwork in this.constructor.supportedNetworks) {
            if (this.constructor.supportedNetworks[supportedNetwork] === label) {
                return supportedNetwork;
            }
        }
        throw `Unknown network ${label}`;
    }

    /**
     * Builds the node with all supported networks
     */
    async build(){
        let node = await super.build(),
            tableView = new ModelTensorRTTableView(
                this.config,
                this.tableData,
                (engineRow) => this.buildEngine(this.getNameFromLabel(engineRow["Network Name"]))
            );

        for (let descriptionPart of this.constructor.tensorRTDescription) {
            node.append(E.p().class("margin").content(descriptionPart));
        }

        return node.append(await tableView.getNode());
    };
};

/**
 * The ModelPickerForm is just the ModelPickerInputView in a form.
 */
class ModelPickerFormView extends FormView {
    /**
     * @var string The class name of the form.
     */
    static className = "model-picker";

    /**
     * @var bool Whether or not to include a submit button
     */
    static autoSubmit = true;

    /**
     * @var object The fieldset legend will be hidden, only one fieldset needed.
     */
    static fieldSets = {
        "Model": {
            "model": {
                "class": ModelPickerInputView
            }
        }
    };
    
    /**
     * @var string The path to the TensorRT logo
     */
    static tensorRTLogo = "/static/img/brand/tensorrt.png";

    /**
     * This is called when the model is changed to display the current TensorRT
     * status, when supported.
     * 
     * @param object $newStatus The new status object, containing at least 'supported' and 'ready' leys
     * @param callable $buildTensorRT The callback function to call when the user clicks a non-ready indicator.
     */
    setTensorRTStatus(newStatus, buildTensorRT) {
        let indicator = this.node.find("#tensorrt");
        if (newStatus.supported) {
            let supportedNetworks = ModelTensorRTStatusView.supportedNetworks,
                supportedNetworkNames = Object.getOwnPropertyNames(supportedNetworks),
                supportedNetworkCount = supportedNetworkNames.length,
                supportedNetworkReady = supportedNetworkNames.reduce((carry, supportedNetwork) => {
                    carry[supportedNetwork] = newStatus[`${supportedNetwork}_ready`];
                    return carry;
                }, {}),
                supportedNetworkReadyCount = Object.values(supportedNetworkReady).filter((value) => value).length;

            if (isEmpty(indicator)) {
                indicator = E.div().id("tensorrt").append(
                    E.img().src(this.constructor.tensorRTLogo),
                    E.span().class("fraction").content(
                        E.span().content(`${supportedNetworkReadyCount}`),
                        E.span().content(`${supportedNetworkCount}`)
                    )
                ).on("click", () => buildTensorRT());
                this.node.append(indicator);
            } else {
                indicator
                    .off("click")
                    .on("click", () => buildTensorRT())
                    .find("span.fraction")
                    .content(
                        E.span().content(`${supportedNetworkReadyCount}`), 
                        E.span().content(`${supportedNetworkCount}`)
                    );
            }

            if (newStatus.ready) {
                indicator.addClass("ready").data("tooltip", "TensorRT is <strong>ready</strong>");
            } else {
                indicator.removeClass("ready").data("tooltip", "TensorRT is <strong>not ready</strong>");
            }

        } else if(!isEmpty(indicator)) {
            this.node.remove(indicator);
        }
    };
};

/**
 * The ModelPickerController appends the model chooser input to the image editor view.
 * It will call the necessary functions to build TensorRT as well.
 */
class ModelPickerController extends Controller {
    /**
     * @var int The width of the TensorRT Status Window
     */
    static tensorRTStatusWindowWidth = 500;
    
    /**
     * @var int The height of the TensorRT Status Window
     */
    static tensorRTStatusWindowHeight = 750;

    /**
     * Get state from the model picker
     */
    getState() {
        return {
            "model": this.formView.values,
            "modelConfig": this.modelConfigurationFormView.values
        };
    }

    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "model": null, 
            "modelConfig": null
        };
    }

    /**
     * Set state in the model picker
     */
    setState(newState) {
        if (!isEmpty(newState.model)) {
            this.formView.suppressDefaults = true;
            this.formView.setValues(newState.model).then(
                () => this.formView.submit()
            );
        }
        if (!isEmpty(newState.modelConfig)) {
            this.modelConfigurationFormView.setValues(newState.modelConfig).then(
                setTimeout(
                    () => this.modelConfigurationFormView.submit(),
                    250
                )
            );
        }
    }

    /**
     * Issues the request to the engine to build a specific engine
     */
    async buildEngine(model, engine) {
        await this.model.post(`/models/${model}/tensorrt/${engine}`);
        this.notify("info", "Build Started", "The engine will be busy throughout this TensorRT build. You will see a notification when it is complete, and the status indicator in the top bar will show ready or idle.");
        await waitFor(
            () => {
                return !isEmpty(this.builtEngines[model]) && this.builtEngines[model].indexOf(engine) !== -1;
            },
            {
                interval: 5000
            }
        );
    }

    /**
     * Build TensorRT for a specified model.
     * @param model.DiffusionModel $model The model from the API
     */
    async showBuildTensorRT(model) {
        let currentStatus = await model.getStatus(),
            currentEngineBuildProcess = await this.getCurrentEngineBuildProcess(),
            tensorRTStatus = {supported: false};

        if (!isEmpty(currentStatus.tensorrt)) {
            tensorRTStatus = currentStatus.tensorrt.base;
            if (!isEmpty(currentStatus.tensorrt.inpainter)) {
                tensorRTStatus.inpaint_unet_ready = currentStatus.tensorrt.inpainter.unet_ready;
            }
        }

        if (!isEmpty(currentEngineBuildProcess) && currentEngineBuildProcess.metadata.tensorrt_build.model === model.name) {
            currentStatus.building = currentEngineBuildProcess.metadata.tensorrt_build.network;
        }
        
        let modelStatusView = new ModelTensorRTStatusView(this.config, tensorRTStatus, (engine) => this.buildEngine(model.name, engine)),
            modelWindow = await this.spawnWindow(
                `${model.name} TensorRT Status`,
                modelStatusView,
                this.constructor.tensorRTStatusWindowWidth,
                this.constructor.tensorRTStatusWindowHeight
            );
        return modelWindow;
    }

    /**
     * Checks if an engine build is currently occurring
     */
    async getCurrentEngineBuildProcess() {
        let currentInvocations = await this.model.get("/invocation");
        for (let invocation of currentInvocations) {
            if (invocation.metadata !== undefined && invocation.metadata.tensorrt_build !== undefined && ["queued", "processing"].indexOf(invocation.status) !== -1) {
                return invocation;
            }
        }
        return null;
    }

    /**
     * When initialized, append form to container and register callbacks.
     */
    async initialize() {
        this.builtEngines = {};
        ModelPickerInputView.defaultOptions = async () => {
            let allModels = await this.model.get("/model-options"),
                modelOptions = allModels.reduce((carry, datum) => {
                    let typeString = datum.type === "checkpoint"
                        ? "Checkpoint"
                        : "Preconfigured Model";
                    carry[`${datum.type}/${datum.name}`] = `<strong>${datum.name}</strong><em>${typeString}</em>`;
                    return carry;
                }, {});
            return modelOptions;
        };

        this.formView = new ModelPickerFormView(this.config);
        this.modelConfigurationFormView = new ModelConfigurationFormView(this.config);

        this.formView.onSubmit(async (values) => {
            let suppressDefaults = this.formView.suppressDefaults;
            this.formView.suppressDefaults = false;
            if (values.model) {
                let [selectedType, selectedName] = values.model.split("/");
                this.engine.model = selectedName;
                this.engine.modelType = selectedType;
                if (selectedType === "model") {
                    this.modelConfigurationFormView.hide();
                    try {
                        let fullModel = await this.model.DiffusionModel.query({name: selectedName}),
                            modelStatus = await fullModel.getStatus(),
                            tensorRTStatus = {supported: false};

                        fullModel.status = modelStatus;
                        if (suppressDefaults) {
                            fullModel._relationships.config = null;
                        }

                        this.publish("modelPickerChange", fullModel);
                        if (!isEmpty(modelStatus.tensorrt)) {
                            tensorRTStatus = modelStatus.tensorrt.base;
                            if (!isEmpty(modelStatus.tensorrt.inpainter)) {
                                tensorRTStatus.inpaint_unet_ready = modelStatus.tensorrt.inpainter.unet_ready;
                            }
                        }
                        this.formView.setTensorRTStatus(
                            tensorRTStatus,
                            () => this.showBuildTensorRT(fullModel)
                        );
                    } catch(e) {
                        // Reset
                        this.formView.setValues({"model": null});
                        console.error(e);
                    }
                } else {
                    this.modelConfigurationFormView.show();
                    this.formView.setTensorRTStatus({supported: false});
                }
            } else {
                this.formView.setTensorRTStatus({supported: false});
            }
        });
        
        this.modelConfigurationFormView.onSubmit(async (values) => {
            this.engine.refiner = values.refiner;
            this.engine.inpainter = values.inpainter;
            this.engine.lora = values.lora;
            this.engine.lycoris = values.lycoris;
            this.engine.inversion = values.inversion;
            this.engine.vae = values.vae;
        });

        this.application.container.appendChild(await this.formView.render());
        this.application.container.appendChild(await this.modelConfigurationFormView.render());

        this.subscribe("invocationError", (payload) => {
            if (!isEmpty(payload.metadata) && !isEmpty(payload.metadata.tensorrt_build)) {
                let network = payload.metadata.tensorrt_build.network,
                    networkName = ModelTensorRTStatusView.supportedNetworks[network],
                    model = payload.metadata.tensorrt_build.model;

                this.notify("info", "TensorRT Engine Build Failed", `${model} ${networkName} TensorRT Engine failed to build. Please try again.`);
            }
        });
        this.subscribe("invocationComplete", (payload) => {
            if (!isEmpty(payload.metadata) && !isEmpty(payload.metadata.tensorrt_build)) {
                let network = payload.metadata.tensorrt_build.network,
                    networkName = ModelTensorRTStatusView.supportedNetworks[network],
                    model = payload.metadata.tensorrt_build.model;

                this.notify("info", "TensorRT Engine Build Complete", `Successfully built ${model} ${networkName} TensorRT Engine.`);
                
                if (isEmpty(this.builtEngines[model])) {
                    this.builtEngines[model] = [];
                }
                this.builtEngines[model].push(network);
            }
        });
    }
}

export { ModelPickerController };
