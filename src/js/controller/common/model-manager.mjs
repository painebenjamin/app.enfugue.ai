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
            "size": "Size",
            "prompt": "Prompt",
            "negative_prompt": "Negative Prompt"
        });
        this.tableView.setSearchFields(["name", "prompt", "negative_prompt", "model"]);
        this.tableView.setFormatter("size", (datum) => `${datum}px`);
        
        // Add the 'Edit' button
        this.tableView.addButton("Edit", "fa-solid fa-edit", async (row) => {
            let modelValues = row.getAttributes();

            modelValues.checkpoint = modelValues.model;
            modelValues.lora = isEmpty(row.lora) ? [] : row.lora.map((lora) => lora.getAttributes());
            modelValues.lycoris = isEmpty(row.lycoris) ? [] : row.lycoris.map((lycoris) => lycoris.getAttributes());
            modelValues.inversion = isEmpty(row.inversion) ? [] : row.inversion.map((inversion) => inversion.model);
            modelValues.vae = isEmpty(row.vae) ? null : row.vae[0].name;

            if (!isEmpty(row.refiner)) {
                modelValues.refiner = row.refiner[0].model;
                modelValues.refiner_size = row.refiner[0].size;
            }
            
            if (!isEmpty(row.inpainter)) {
                modelValues.inpainter = row.inpainter[0].model;
                modelValues.inpainter_size = row.inpainter[0].size;
            }

            if (!isEmpty(row.config)) {
                let defaultConfig = {};
                for (let configItem of row.config) {
                    defaultConfig[configItem.configuration_key] = configItem.configuration_value;
                }

                modelValues = {...modelValues, ...defaultConfig};

                if (!isEmpty(defaultConfig.prompt_2)) {
                    modelValues.prompt = [modelValues.prompt, defaultConfig.prompt_2];
                }
                if (!isEmpty(defaultConfig.negative_prompt_2)) {
                    modelValues.negative_prompt = [defaultConfig.negative_prompt, defaultConfig.negative_prompt_2];
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_prompt_2)) {
                    modelValues.upscale_diffusion_prompt = defaultConfig.upscale_diffusion_prompt.map(
                        (prompt, index) => [prompt, defaultConfig.upscale_diffusion_prompt_2[index]],
                    );
                }
                if (!isEmpty(defaultConfig.upscale_diffusion_negative_prompt_2)) {
                    modelValues.upscale_diffusion_negative_prompt = defaultConfig.upscale_diffusion_negative_prompt.map(
                        (prompt, index) => [prompt, defaultConfig.upscale_diffusion_negative_prompt_2[index]],
                    );
                }
            }

            if (!isEmpty(row.scheduler)) {
                modelValues.scheduler = row.scheduler[0].name;
            }

            let modelForm = new ModelFormView(this.config, deepClone(modelValues)),
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
                }
                if (Array.isArray(updatedValues.negative_prompt)) {
                    updatedValues.negative_prompt_2 = updatedValues.negative_prompt[1];
                    updatedValues.negative_prompt = updatedValues.negative_prompt[0];
                }
                let upscalePrompt = [],
                    upscalePrompt2 = [],
                    upscaleNegativePrompt = [],
                    upscaleNegativePrompt2 = [];
                
                if (!isEmpty(updatedValues.upscale_diffusion_prompt)) {
                    for (let promptPart of updatedValues.upscale_diffusion_prompt) {
                        if (Array.isArray(promptPart)) {
                            upscalePrompt.push(promptPart[0]);
                            upscalePrompt2.push(promptPart[1]);
                        } else {
                            upscalePrompt.push(promptPart);
                            upscalePrompt2.push(null);
                        }
                    }
                }
                if (!isEmpty(updatedValues.upscale_diffusion_negative_prompt)) {
                    for (let promptPart of updatedValues.upscale_diffusion_negative_prompt) {
                        if (Array.isArray(promptPart)) {
                            upscaleNegativePrompt.push(promptPart[0]);
                            upscaleNegativePrompt2.push(promptPart[1]);
                        } else {
                            upscaleNegativePrompt.push(promptPart);
                            upscaleNegativePrompt2.push(null);
                        }
                    }
                }

                updatedValues.upscale_diffusion_prompt = upscalePrompt;
                updatedValues.upscale_diffusion_prompt_2 = upscalePrompt2;
                updatedValues.upscale_diffusion_negative_prompt = upscaleNegativePrompt;
                updatedValues.upscale_diffusion_negative_prompt_2 = upscaleNegativePrompt2;

                try {
                    await this.model.patch(`/models/${row.name}`, null, null, updatedValues);
                    if (!isEmpty(modelWindow)) {
                        modelWindow.remove();
                    }
                    this.tableView.requery();
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
                `Edit ${row.name}`,
                modelForm,
                this.constructor.modelWindowWidth,
                this.constructor.modelWindowHeight
            );
        });
        
        // Add the 'Delete' button
        this.tableView.addButton("Delete", "fa-solid fa-trash", async (row) => {
            try{
                await this.model.delete(`/models/${row.name}`);
                this.tableView.requery();
            } catch(e) {
                let errorMessage = isEmpty(e)
                    ? "Couldn't communicate with server."
                    : isEmpty(e.detail)
                        ? `${e}`
                        : e.detail;

                this.notify("error", "Couldn't delete model", errorMessage);
                modelForm.enable();
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
            let upscalePrompt = [],
                upscalePrompt2 = [],
                upscaleNegativePrompt = [],
                upscaleNegativePrompt2 = [];
            
            if (!isEmpty(values.upscale_diffusion_prompt)) {
                for (let promptPart of values.upscale_diffusion_prompt) {
                    if (Array.isArray(promptPart)) {
                        upscalePrompt.push(promptPart[0]);
                        upscalePrompt2.push(promptPart[1]);
                    } else {
                        upscalePrompt.push(promptPart);
                        upscalePrompt2.push(null);
                    }
                }
            }
            if (!isEmpty(values.upscale_diffusion_negative_prompt)) {
                for (let promptPart of values.upscale_diffusion_negative_prompt) {
                    if (Array.isArray(promptPart)) {
                        upscaleNegativePrompt.push(promptPart[0]);
                        upscaleNegativePrompt2.push(promptPart[1]);
                    } else {
                        upscaleNegativePrompt.push(promptPart);
                        upscaleNegativePrompt2.push(null);
                    }
                }
            }

            values.upscale_diffusion_prompt = upscalePrompt;
            values.upscale_diffusion_prompt_2 = upscalePrompt2;
            values.upscale_diffusion_negative_prompt = upscaleNegativePrompt;
            values.upscale_diffusion_negative_prompt_2 = upscaleNegativePrompt2;

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
