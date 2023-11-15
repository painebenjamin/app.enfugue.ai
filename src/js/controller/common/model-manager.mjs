/** @module controller/common/model-manager */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { ParentView } from "../../view/base.mjs";
import { TableView, ModelTableView } from "../../view/table.mjs";
import { ButtonInputView } from "../../forms/input.mjs";
import { ModelFormView } from "../../forms/enfugue/models.mjs";

/**
 * New model input is a simple extension of the buttonview
 */
class NewModelInputView extends ButtonInputView {
    /**
     * @var string The value to display in the button
     */
    static defaultValue = "New Model Configuration";

    /**
     * @var string The class name for CSS
     */
    static className = "new-model-input-view";
};

/**
 * This shared controller makes it easy to view/edit 
 * models from anywhere else in the application.
 */
class ModelManagerController extends Controller {
    /**
     * @var int The starting width of the model form window
     */
    static modelWindowWidth = 400;
    
    /**
     * @var int The starting height of the model form window
     */
    static modelWindowHeight = 1000;

    /**
     * @var int The starting width of the manager form window
     */
    static managerWindowWidth = 800;
    
    /**
     * @var int The starting height the manager form window
     */
    static managerWindowHeight = 600;

    getPayloadFromModel(model, usePromptArrays = true) {
        let modelValues = model.getAttributes();

        modelValues.checkpoint = modelValues.model;
        modelValues.lora = isEmpty(model.lora) ? [] : model.lora.map((lora) => lora.getAttributes());
        modelValues.lycoris = isEmpty(model.lycoris) ? [] : model.lycoris.map((lycoris) => lycoris.getAttributes());
        modelValues.inversion = isEmpty(model.inversion) ? [] : model.inversion.map((inversion) => inversion.model);
        modelValues.vae = isEmpty(model.vae) ? null : model.vae[0].name;
        modelValues.motion_module = isEmpty(model.motion_module) ? null : model.motion_module[0].name;

        if (!isEmpty(model.refiner)) {
            modelValues.refiner = model.refiner[0].model;
        }
        
        if (!isEmpty(model.inpainter)) {
            modelValues.inpainter = model.inpainter[0].model;
        }

        if (!isEmpty(model.config)) {
            let defaultConfig = {};
            for (let configItem of model.config) {
                defaultConfig[configItem.configuration_key] = configItem.configuration_value;
            }

            modelValues = {...modelValues, ...defaultConfig};

            if (!isEmpty(defaultConfig.prompt_2) && usePromptArrays) {
                modelValues.prompt = [modelValues.prompt, defaultConfig.prompt_2];
            }
            if (!isEmpty(defaultConfig.negative_prompt_2) && usePromptArrays) {
                modelValues.negative_prompt = [modelValues.negative_prompt, defaultConfig.negative_prompt_2];
            }
        }

        if (!isEmpty(model.scheduler)) {
            modelValues.scheduler = model.scheduler[0].name;
        }

        return deepClone(modelValues);
    }

    /**
     * Creates a window to edit a configuration
     */
    async showEditModel(model) {
        let modelValues = this.getPayloadFromModel(model),
            modelForm = new ModelFormView(this.config, modelValues),
            modelWindow;

        modelForm.onChange(async (updatedValues) => {
            if (!isEmpty(modelForm.values.refiner)) {
                modelForm.addClass("show-refiner");
            } else {
                modelForm.removeClass("show-refiner");
            }
            if (!isEmpty(modelForm.values.inpainter)) {
                modelForm.addClass("show-inpainter");
            } else {
                modelForm.removeClass("show-inpainter");
            }
        });

        modelForm.onSubmit(async (updatedValues) => {
            if (Array.isArray(updatedValues.prompt)) {
                updatedValues.prompt_2 = updatedValues.prompt[1];
                updatedValues.prompt = updatedValues.prompt[0];
            } else {
                updatedValues.prompt_2 = null;
            }
            if (Array.isArray(updatedValues.negative_prompt)) {
                updatedValues.negative_prompt_2 = updatedValues.negative_prompt[1];
                updatedValues.negative_prompt = updatedValues.negative_prompt[0];
            } else {
                updatedValues.negative_prompt_2 = null;
            }

            try {
                await this.model.patch(`/models/${model.name}`, null, null, updatedValues);
                if (!isEmpty(modelWindow)) {
                    modelWindow.remove();
                }
                if (!isEmpty(this.tableView)) {
                    this.tableView.requery();
                }
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail)
                        ? `${e}`
                        : e.detail;

                this.notify("error", "Couldn't update model", errorMessage);
                modelForm.enable();
            }
        });
        modelForm.onCancel(() => modelWindow.remove());
        modelWindow = await this.spawnWindow(
            `Edit ${model.name}`,
            modelForm,
            this.constructor.modelWindowWidth,
            this.constructor.modelWindowHeight
        );
    }

    /**
     * Creates the manager table.
     */
    async createManager() {
        this.tableView = new ModelTableView(this.config, this.model.DiffusionModel);
        this.buttonView = new NewModelInputView(this.config);

        // Set columns, formatters, searchables
        this.tableView.setColumns({
            "name": "Name",
            "model": "Model",
            "prompt": "Prompt",
            "negative_prompt": "Negative Prompt"
        });
        this.tableView.setSearchFields(["name", "prompt", "negative_prompt", "model"]);
        this.tableView.setFormatter("size", (datum) => `${datum}px`);
        
        // Add the 'Edit' button
        this.tableView.addButton("Edit", "fa-solid fa-edit", (row) => this.showEditModel(row));
        
        // Add the 'Copy' button
        this.tableView.addButton("Copy", "fa-solid fa-copy", async (row) => {
            try {
                let payload = this.getPayloadFromModel(row, false);
                payload.name = `${payload.name} (Copy)`;
                let response = await this.model.post("/models", null, null, payload);
                await this.tableView.requery();
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail)
                        ? `${e}`
                        : e.detail;

                this.notify("error", "Couldn't delete model", errorMessage);
            }
        });

        // Add the 'Delete' button
        this.tableView.addButton("Delete", "fa-solid fa-trash", async (row) => {
            try {
                await this.model.delete(`/models/${row.name}`);
                this.tableView.requery();
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail)
                        ? `${e}`
                        : e.detail;

                this.notify("error", "Couldn't delete model", errorMessage);
            }
        });

        this.buttonView.onChange(() => {
            this.showNewModel();
        });

        let managerView = new ParentView(this.config);
        managerView.addChild(this.tableView);
        managerView.addChild(this.buttonView);
        return managerView;
    }

    /**
     * Creates the 'New Model' form.
     */
    async createModelForm() {
        let modelForm = new ModelFormView(this.config);
        modelForm.onSubmit(async (values) => {
            if (Array.isArray(values.prompt)) {
                values.prompt_2 = values.prompt[1];
                values.prompt = values.prompt[0];
            }
            if (Array.isArray(values.negative_prompt)) {
                values.negative_prompt_2 = values.negative_prompt[1];
                values.negative_prompt = values.negative_prompt[0];
            }

            try {
                let response = await this.model.post("/models", null, null, values);
                if (!isEmpty(this.newModelWindow)) {
                    this.newModelWindow.remove();
                    this.newModelWindow = null;
                }
                if (!isEmpty(this.managerWindow) && !isEmpty(this.tableView)) {
                    this.tableView.requery();
                }
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail) 
                        ? `${e}` 
                        : e.detail;
                this.notify("Error", "Couldn't create model", errorMessage);
                modelForm.enable();
            }
        });
        modelForm.onChange(async (updatedValues) => {
            if (!isEmpty(modelForm.values.refiner)) {
                modelForm.addClass("show-refiner");
            } else {
                modelForm.removeClass("show-refiner");
            }
            if (!isEmpty(modelForm.values.inpainter)) {
                modelForm.addClass("show-inpainter");
            } else {
                modelForm.removeClass("show-inpainter");
            }
        });
        modelForm.onCancel(() => {
            this.newModelWindow.remove();
            this.newModelWindow = null;
        });
        return modelForm;
    }
    
    /**
     * Shows the New Model form.
     */
    async showNewModel() {
        if (!isEmpty(this.newModelWindow)) {
            this.newModelWindow.focus();
        } else {
            this.newModelWindow = await this.spawnWindow(
                "New Configuration",
                this.createModelForm(),
                this.constructor.modelWindowWidth,
                this.constructor.modelWindowHeight
            );
            this.newModelWindow.onClose(() => { delete this.newModelWindow; });
        }
    }

    /**
     * Shows the model manager table.
     */
    async showManager() {
        if (!isEmpty(this.managerWindow)) {
            this.managerWindow.focus();
        } else {
            this.managerWindow = await this.spawnWindow(
                "Model Configuration Manager",
                this.createManager(),
                this.constructor.managerWindowWidth,
                this.constructor.managerWindowHeight
            );
            this.managerWindow.onClose(() => { delete this.managerWindow; });
        }
    }
}

export { ModelManagerController };
