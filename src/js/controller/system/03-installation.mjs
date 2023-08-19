/** @module controller/system/03-installation */
import { isEmpty, humanSize } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { MenuController } from "../menu.mjs";
import { View, ParentView } from "../../view/base.mjs";
import { TableView } from "../../view/table.mjs";
import { ButtonInputView } from "../../forms/input.mjs";
import { DirectoryFormView, FileFormView } from "../../forms/enfugue/files.mjs";

const E = new ElementBuilder();

/**
 * This extends the button input view to set css classes and content
 */
class UploadFileButtonInputView extends ButtonInputView {
    /**
     * @var string custom classname
     */
    static className = "upload-file-input-view";

    /**
     * @var string custom button content
     */
    static defaultValue = "Upload File";
}

/**
 * This is the table view for the summary over all directories
 */
class InstallationDirectorySummaryTableView extends TableView {
    /**
     * @var string Custom class name
     */
    static className = "installation-directory-summary-table-view";

    /**
     * @var object column names and labels
     */
    static columns = {
        "location": "Location",
        "directory": "Directory",
        "items": "Items",
        "bytes": "Total File Size"
    };

    /**
     * @var object column format functions
     */
    static columnFormatters = {
        "bytes": (bytes) => humanSize(bytes)
    };
};

/**
 * This is the table view for a single directory
 */
class InstallationDirectoryTableView extends TableView {
    /**
     * @var string Custom class name
     */
    static className = "installation-directory-table-view";

    /**
     * @var object column names and labels
     */
    static columns = {
        "name": "Name",
        "bytes": "Total File Size"
    };

    /**
     * @var object column format functions
     */
    static columnFormatters = {
        "bytes": (bytes) => humanSize(bytes)
    };
};

/**
 * This is the table view for all TRT engines
 */
class TensorRTEngineTableView extends TableView {
    /**
     * @var object column names and labels
     */
    static columns = {
        "model": "Model",
        "type": "Type",
        "size": "Size",
        "lora": "LoRA",
        "inversion": "Inversions",
        "bytes": "File Size",
        "used_by": "Used By"
    };

    /**
     * @var object column format functions
     */
    static columnFormatters = {
        "type": (typename, row) => {
            let prefix = "";
            if (row.model.endsWith("-inpainting")) {
                prefix = "Inpainting ";
            }
            let suffix = {
                "unet": "UNet",
                "controlledunet": "Controlled UNet",
                "clip": "CLIP", // Unused
                "vae": "VAE", // Unused
                "controlnet": "ControlNet" // Unused
            }[typename];
            return `${prefix}${suffix}`;
        },
        "size": (size) => `${size}px`,
        "bytes": (bytes) => humanSize(bytes),
        "lora": (lora) => isEmpty(lora) ? "None" : lora.map((loraPart) => loraPart.join(":")).join("<br />"),
        "inversion": (inversion) => isEmpty(inversion) ? "None" : inversion.map((inversionPart) => inversionPart.model).join("<br />"),
        "used_by": (usedBy) => isEmpty(usedBy) ? "None" : usedBy.length <= 2 ? usedBy.join("<br />") : usedBy.slice(0, 2).join("<br />") + `<br />…and ${usedBy.length-2} more`
    };
};

/**
 * This is the table view for TRT engines; this will be a single row always
 */
class TensorRTEngineSummaryTableView extends TableView {
    /**
     * @var object column names and labels
     */
    static columns = {
        "total": "Total Engines",
        "used": "Currently Used",
        "bytes": "Total File Size"
    };

    /**
     * @var object column format functions
     */
    static columnFormatters = {
        "total": (total) => isEmpty(total) ? "None" : `${total}`,
        "used": (used) => isEmpty(used) ? "None" : `${used}`,
        "bytes": (bytes) => isEmpty(bytes) ? "0 Bytes" : humanSize(bytes)
    };
};

/**
 * The SummaryView shows both summary tables, headers, and binds callbacks
 */
class InstallationSummaryView extends View {
    /**
     * Override constructor to pass the controller itself
     */
    constructor(controller) {
        super(controller.config);
        this.controller = controller;

        this.summaryTable = new InstallationDirectorySummaryTableView(this.config);
        this.summaryTable.addButton("Manage", "fa-solid fa-list-check", (row) => {
            this.controller.showDirectoryManager(row.directory);
        });
        this.summaryTable.addButton("Change Directory", "fa-solid fa-edit", (row) => {
            this.controller.showChangeDirectory(row.directory, row.location);
        });
        this.engineTable = new TensorRTEngineSummaryTableView(this.config);
        this.engineTable.addButton("Manage", "fa-solid fa-list-check", () => {
            this.controller.showTensorRTManager();
        });
    }

    /**
     * Call to update the table when it is visible
     */
    async update() {
        let installationSummary = await this.controller.model.get("/installation"),
            tensorrtSummary = await this.controller.model.get("/tensorrt");
        
        await this.summaryTable.setData(Object.getOwnPropertyNames(installationSummary).map((directory) => {
            return {
                "directory": directory,
                "location": installationSummary[directory].path,
                "items": installationSummary[directory].items,
                "bytes": installationSummary[directory].bytes
            };
        }), false);

        await this.engineTable.setData([tensorrtSummary.reduce((carry, item) => {
            if (carry.total === undefined) carry.total = 0;
            if (carry.used === undefined) carry.used = 0;
            if (carry.bytes === undefined) carry.bytes = 0;
            carry.total += 1;
            carry.used += item.used;
            carry.bytes += item.bytes;
            return carry;
        }, {})], false);
    }

    /**
     * On build, first update, then add tables
     */
    async build() {
        let node = await super.build();
        await this.update();
        return node.content(
            E.h2().content("Weights and Configuration"),
            await this.summaryTable.getNode(),
            E.h2().content("TensorRT Engines"),
            await this.engineTable.getNode()
        );
    }
}

/**
 * The systems setting controll just opens up the system settings form(s)
 */
class InstallationController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Installation";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-folder-tree";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "i";

    /**
     * @var int The width of the summary window
     */
    static summaryWindowWidth = 600;

    /**
     * @var int The height of the summary window
     */
    static summaryWindowHeight = 600;
    
    /**
     * @var int The width of the manager window
     */
    static managerWindowWidth = 600;

    /**
     * @var int The height of the manager window
     */
    static managerWindowHeight = 800;
    
    /**
     * @var int The width of the upload window
     */
    static uploadWindowWidth = 400;

    /**
     * @var int The height of the upload window
     */
    static uploadWindowHeight = 250;

    /**
     * @var int The width of the change directory form
     */
    static changeDirectoryWindowWidth = 400;

    /**
     * @var int The height of the change directory form
     */
    static changeDirectoryWindowHeight = 200;

    /**
     * @var array The directories that can be uploaded to
     */
    static uploadableDirectories = ["lora", "checkpoint", "inversion", "lycoris"];

    /**
     * On initialize, add local object to maintain window state
     */
    async initialize() {
        await super.initialize();
        this.directoryWindows = {};
    }

    /**
     * Shows the window of the installation summary
     */
    async showSummaryWindow() {
        if (!isEmpty(this.summaryWindow)) {
            this.summaryWindow.focus();
        } else {
            this.summaryWindow = await this.spawnWindow(
                "Installation Summary",
                await this.getSummaryView(),
                this.constructor.summaryWindowWidth,
                this.constructor.summaryWindowHeight
            );
            this.summaryWindow.onClose(() => { this.summaryWindow = null; });
        }
    }

    /**
     * Gets the overall summary view
     */
    async getSummaryView() {
        if (isEmpty(this.summaryView)) {
            this.summaryView = new InstallationSummaryView(this);
        } else {
            this.summaryView.update();
        }
        return this.summaryView;
    }

    /**
     * Shows the manager just for TensorRT
     */
    async showTensorRTManager() {
        let tensorRTData = await this.model.get("/tensorrt");
        if (isEmpty(this.tensorRTManagerWindow)) {
            let table = new TensorRTEngineTableView(this.config, tensorRTData);
            table.addButton("Delete", "fa-solid fa-trash", async (row) => {
                let url = `/tensorrt/${row.model}/${row.type}/${row.key}`;
                try {
                    await this.model.delete(url);
                    this.notify("info", "Success", "TensorRT engine deleted.");
                    table.setData(await this.model.get("/tensorrt"), false);
                    if (!isEmpty(this.summaryView)) {
                        this.summaryView.update();
                    }
                } catch(e) {
                    let errorMessage = `${e}`;
                    if (!isEmpty(e.detail)) errorMessage = e.detail;
                    else if (!isEmpty(e.title)) errorMessage = e.title;
                    this.notify("error", "Couldn't Delete", errorMessage);
                }
            });
            this.tensorRTManagerWindow = await this.spawnWindow(
                "TensorRT Engines",
                table,
                this.constructor.managerWindowWidth,
                this.constructor.managerWindowHeight
            );
            this.tensorRTManagerWindow.onClose(() => { this.tensorRTManagerWindow = null; });
        } else {
            this.tensorRTManagerWindow.content.setData(tensorRTData, false);
            this.tensorRTManagerWindow.focus();
        }
    }

    /**
     * Shows the 'upload file' dialogue for select directories
     */
    async showUploadFile(directory) {
        let uploadFileForm = new FileFormView(this.config),
            uploadFileWindow = await this.spawnWindow(
                `Upload File to '${directory}'`,
                uploadFileForm,
                this.constructor.uploadWindowWidth,
                this.constructor.uploadWindowHeight
            );
        uploadFileForm.onSubmit(async (formValues) => { 
            await this.model.multiPart(
                `/installation/${directory}`,
                null,
                null, 
                { file: formValues.file },
                async (progressEvent) => {
                    (await uploadFileForm.getInputView("file")).setProgress(
                        progressEvent.loaded/progressEvent.total
                    );
                }
            );
            this.notify("info", "Uploaded", `File successfully uploaded to '${directory}'`);
            if (!isEmpty(this.summaryView)) {
                this.summaryView.update();
            }
            if (!isEmpty(this.directoryWindows[directory])) {
                this.directoryWindows[directory].content.getChild(0).setData(
                    await this.model.get(`/installation/${directory}`),
                    false
                );
            }
            uploadFileWindow.remove();
        });
        uploadFileForm.onCancel(() => { uploadFileWindow.remove(); });
    }

    /**
     * Shows the 'change directory' dialogue for a directory
     */
    async showChangeDirectory(directory, currentValue) {
        let changeDirectoryForm = new DirectoryFormView(this.config, {"directory": currentValue}),
            changeDirectoryWindow = await this.spawnWindow(
                `Change Filesystem Location for ${directory}`,
                changeDirectoryForm,
                this.constructor.changeDirectoryWindowWidth,
                this.constructor.changeDirectoryWindowHeight
            );

        changeDirectoryForm.onSubmit(async (formValues) => {
            try {
                await this.model.post(
                    `/installation/${directory}/move`,
                    null,
                    null,
                    { directory: formValues.directory }
                );
                this.notify("info", "Changed", `Filesystem Location successfully changed for ${directory}`);
                if (!isEmpty(this.summaryView)) {
                    this.summaryView.update();
                }
                if (!isEmpty(this.directoryWindows[directory])) {
                    this.directoryWindows[directory].content.getChild(0).setData(
                        await this.model.get(`/installation/${directory}`),
                        false
                    );
                }
                changeDirectoryWindow.remove();
            } catch(e) {
                let errorMessage = `${e}`;
                if (!isEmpty(e.detail)) errorMessage = e.detail;
                else if (!isEmpty(e.title)) errorMessage = e.title;
                this.notify("error", "Couldn't Change", errorMessage);
                changeDirectoryForm.enable();
            }
        });
        changeDirectoryForm.onCancel(() => { changeDirectoryWindow.remove(); });
    }

    /**
     * Shows the manager for a directory
     */
    async showDirectoryManager(directory) {
        let directoryData = await this.model.get(`/installation/${directory}`);
        if (isEmpty(this.directoryWindows[directory])) {
            let table = new InstallationDirectoryTableView(this.config, directoryData);
            table.addButton("Delete", "fa-solid fa-trash", async (row) => {
                let url = `/installation/${directory}/${row.name}`;
                try {
                    await this.model.delete(url);
                    this.notify("info", "Success", `Successfully deleted <strong>${directory}/${row.name}</strong>`);
                    table.setData(await this.model.get(`/installation/${directory}`), false);
                    if (!isEmpty(this.summaryView)) {
                        this.summaryView.update();
                    }
                } catch(e) {
                    let errorMessage = `${e}`;
                    if (!isEmpty(e.detail)) errorMessage = e.detail;
                    else if (!isEmpty(e.title)) errorMessage = e.title;
                    this.notify("error", "Couldn't Delete", errorMessage);
                }
            });
            let container = new ParentView(this.config);
            container.addClass("installation-summary-view");
            container.addChild(table);
            if (this.constructor.uploadableDirectories.indexOf(directory) !== -1) {
                let uploadView = new UploadFileButtonInputView(this.config);
                uploadView.onChange(() => {
                    this.showUploadFile(directory);
                });
                container.addChild(uploadView);
            }
            this.directoryWindows[directory] = await this.spawnWindow(
                `Manage Directory '${directory}'`,
                container,
                this.constructor.managerWindowWidth,
                this.constructor.managerWindowHeight
            );
            this.directoryWindows[directory].onClose(() => { this.directoryWindows[directory] = null; });
        } else {
            this.directoryWindows[directory].content.getChild(0).setData(directoryData);
            this.directoryWindows[directory].focus();
        }
    }

    /**
     * When clicked, show summary window.
     */
    async onClick() {
        this.showSummaryWindow();
    }
};

export { InstallationController as MenuController };
