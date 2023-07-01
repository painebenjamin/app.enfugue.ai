/** @module controller/common/module-picker */
import { Controller } from "../base.mjs";
import { TableView } from "../../view/table.mjs";
import { View } from "../../view/base.mjs";
import { FormView } from "../../view/forms/base.mjs";
import {
    SearchListInputView, 
    StringInputView, 
    SearchListInputListView
} from "../../view/forms/input.mjs";
import { MultiLoraInputView, MultiInversionInputView } from "./model-manager.mjs";
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
 * This form allows additional pipeline weights when using a checkpoint
 */
class AdditionalWeightsFormView extends FormView {
    /**
     * @var string Custom CSS class
     */
    static className = "additional-weights-form-view";

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
        "Additional Weights": {
            "lora": {
                "class": MultiLoraInputView,
                "label": "LoRA",
                "config": {
                    "tooltip": "LoRA stands for <strong>Low Rank Adapation</strong>, it is a kind of fine-tuning that can perform very specific modifications to Stable Diffusion such as training an individual's appearance, new products that are not in Stable Diffusion's training set, etc."
                }
            },
            "inversion": {
                "class": MultiInversionInputView,
                "label": "Textual Inversion",
                "config": {
                    "tooltip": "Textual Inversion is another kind of fine-tuning that teaches novel concepts to Stable Diffusion in a small number of images, which can be used to positively or negatively affect the impact of various prompts."
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
        "The compiled engine is only useable for a model with the same checkpoint, LoRA, Textual Inversion and engine size. If you change any of those details about this model, it will require recompilation. You can safely change model prompts as desired without requiring a new engine.",
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
        return { "model": this.formView.values, "weights": this.additionalWeightsFormView.values };
    }

    /**
     * Gets default state
     */
    getDefaultState() {
        return { "model": null, "weights": null };
    }

    /**
     * Set state in the model picker
     */
    setState(newState) {
        if (!isEmpty(newState.model)) {
            this.formView.setValues(newState.model).then(() => this.formView.submit());
        }
        if (!isEmpty(newState.weights)) {
            this.additionalWeightsFormView.setValues(newState.weights).then(() => this.additionalWeightsFormView.submit());
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
        let currentStatus = await model.getTensorRTStatus(),
            currentEngineBuildProcess = await this.getCurrentEngineBuildProcess();

        if (!isEmpty(currentEngineBuildProcess) && currentEngineBuildProcess.metadata.tensorrt_build.model === model.name) {
            currentStatus.building = currentEngineBuildProcess.metadata.tensorrt_build.network;
        }
        
        let modelStatusView = new ModelTensorRTStatusView(this.config, currentStatus, (engine) => this.buildEngine(model.name, engine)),
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
                    carry[`${datum.type}/${datum.name}`] = `<strong>${datum.name}</strong><em>(${datum.type})</strong>`;
                    return carry;
                }, {});
            return modelOptions;
        };

        this.formView = new ModelPickerFormView(this.config);
        this.additionalWeightsFormView = new AdditionalWeightsFormView(this.config);

        this.formView.onSubmit(async (values) => {
            if (values.model) {
                let [selectedType, selectedName] = values.model.split("/");
                this.engine.model = selectedName;
                this.engine.modelType = selectedType;
                if (selectedType === "model") {
                    this.additionalWeightsFormView.hide();
                    try {
                        let fullModel = await this.model.DiffusionModel.query({name: selectedName}),
                            tensorRTStatus = await fullModel.getTensorRTStatus();
                        this.publish("modelPickerChange", fullModel);
                        this.formView.setTensorRTStatus(
                            tensorRTStatus, 
                            () => this.showBuildTensorRT(fullModel)
                        );
                    } catch(e) {
                        // Reset
                        this.formView.setValues({"model": null});
                    }
                } else {
                    this.additionalWeightsFormView.show();
                    this.formView.setTensorRTStatus({supported: false});
                }
            } else {
                this.formView.setTensorRTStatus({supported: false});
            }
        });
        
        this.additionalWeightsFormView.onSubmit(async (values) => {
            this.engine.lora = values.lora;
            this.engine.inversion = values.inversion;
        });

        this.application.container.appendChild(await this.formView.render());
        this.application.container.appendChild(await this.additionalWeightsFormView.render());

        this.subscribe("invocationError", (payload) => {
            console.error(payload);
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
