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

        this.tableView.setColumns({
            "name": "Name",
            "model": "Model",
            "size": "Size",
            "prompt": "Prompt",
            "negative_prompt": "Negative Prompt"
        });
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
                for (let defaultConfig of row.config) {
                    modelValues[defaultConfig.configuration_key] = defaultConfig.configuration_value;
                }
            }

            if (!isEmpty(row.scheduler)) {
                for (let scheduler of row.scheduler) {
                    if (scheduler.context === "multi_diffusion") {
                        modelValues.multi_scheduler = scheduler.name;
                    } else {
                        modelValues.scheduler = scheduler.name;
                    }
                }
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
