/** @module controller/common/announcements */
import {
    isEmpty,
    waitFor,
    humanSize,
    humanDuration,
    titleCase
} from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { View, ParentView } from "../../view/base.mjs";
import { TableView } from "../../view/table.mjs";
import { FormView } from "../../forms/base.mjs";
import { ButtonInputView, StringInputView } from "../../forms/input.mjs";

const E = new ElementBuilder();

/**
 * Extends the ButtonInputView for custom CSS and text
 */
class AcknowledgeButtonInputView extends ButtonInputView {
    /**
     * @var string CSS class
     */
    static className = "download-input-view";

    /**
     * @var string The text to show
     */
    static defaultValue = "I Acknowledge";
}

/**
 * This class shows the initial message display to the user with privacy and terms
 */
class InitializationAnnouncementView extends View {
    /**
     * The constructor receives the initial directory structure.
     */
    constructor(config, directories) {
        super(config);
        this.directories = directories;
        this.formView = new FormView(config);
        for (let directory in this.directories) {
            this.formView.addInput("Directories", StringInputView, directory, titleCase(directory), { "required": true, "value": this.directories[directory] });
        }
    }

    /**
     * On build, append text and links
     */
    async build() {
        let node = await super.build();
        node.append(E.h2().content("Welcome to Enfugue"));
        node.append(E.p().content("Thank you for downloading Enfugue! I hope you enjoy it. Please read the following brief disclaimers before you begin."));
        node.append(E.h3().content("Your Privacy"));
        node.append(E.p().content("Enfugue does not track usage in any way. Your images, prompts, configurations, and all other details only exist on your computer and are never sent elsewhere over a network or otherwise, unless specifically requested by the user."));
        node.append(
            E.p().content(
                E.span().content("Some resources, such as public model data, is provided by "),
                E.a().content("Hugging Face").href("https://huggingface.co/").target("_blank"),
                E.span().content(" and by using this application Enfugue will reach out to their servers to download resources as needed. By doing so, you agree to their "),
                E.a().content("Privacy Policy").href("https://huggingface.co/privacy").target("_blank"),
                E.span().content(".")
            )
        );
        node.append(E.p().content("Additional resources may be downloaded from other providers at your request. Please review the privacy policy and terms of service of the resource providers stated when these resources are requested."));
        node.append(E.h3().content("Terms of Service"));
        node.append(E.p().content("Enfugue makes no guarantees of this software's functionality, and by continuing you release Enfugue, Benjamin Paine, GitHub, Hugging Face, Stability AI, Runway, CivitAI, and all other parties whose software contributions are employed within Enfugue from any liability of any kind arising from use of the software, including but not limited to wear and tear on your hardware and expenses incurred by consuming bandwidth."))
        node.append(E.p().content("Furthermore, you hold all the above parties blameless from any and all liability, damage, loss, and expense (including without limitation reasonable attorney’s fees and court costs) arising from claims against parties that Enfugue, or your use of the same as permitted by this Agreement, infringe the intellectual property rights of a third party."));
        node.append(E.p().class("strong").content("Enfugue is and always will be free and open-source software."));
        node.append(E.p().content("To support further development, please consider financially supporting it if you are able. Simply click 'About' under 'Help' above to see ways you can help keep Enfugue improving."));
        node.append(E.h2().content("Directories"));
        node.append(E.p().content("Use these fields to specify the directories to use for your installation. These do not need to be solely used by Enfugue, and Enfugue will never delete anything from these directories unless you specifically request."));
        node.append(await this.formView.getNode());
        return node;
    }
};

/**
 * This is another pure text class just for giving a preamble on downloads
 */
class DownloadAnnouncementHeaderView extends View {
    /**
     * On build, add text
     */
    async build(){
        let node = await super.build();
        node.content(
            E.h2().content("Downloads"),
            E.p().content("The following downloads are required by Enfugue to function. They are the base Stable Diffusion models, from which more fine-tuned models are made. These models are used when you are not using any other models, as well as when calculating derived models."),
            E.p().content("Click the icon in each row to being downloading them. If you do not download them now, they will be downloaded the first time they are needed.")
        );
        return node;
    }
};

/**
 * This is another pure text class just for giving a preamble on SDXL downloads
 */
class OptionalDownloadAnnouncementHeaderView extends View {
    /**
     * On build, add text
     */
    async build(){
        let node = await super.build();
        node.content(
            E.h2().content("Optional Downloads"),
            E.p().content("The following downloads are optional. They are newer, more powerful versions of Stable Diffusion that can be used in Enfugue."),
            E.p().content("For Stable Diffusion XL, you will require a GPU with at least 12 GB of VRAM."),
            E.p().content("Click the icon in each row to being downloading them. You can also select the appropriate checkpoint from the checkpoint picker at any time, and the file will be downloaded if it has not yet been.")
        );
        return node;
    }
};

/**
 * Extend the table class to disable sorting and set columns
 */
class DownloadTableView extends TableView {
    /**
     * @var bool disable sorting
     */
    static canSort = false;

    /**
     * @var object columns and labels
     */
    static columns = {
        "model": "Model",
        "size": "Size",
    };
    
    /**
     * @var object Format the size to better units
     */
    static columnFormatters = {
        "size": (size) => humanSize(size)
    };
}

/**
 * This class shows above the table describing updates
 */
class UpdateAnnouncementHeaderView extends View {
    /**
     * @var string The link ot the releases page
     */
    static releasesLink = "https://github.com/painebenjamin/app.enfugue.ai/releases/";

    /**
     * On build, append text and links
     */
    async build() {
        let node = await super.build();
        node.content(
            E.h2().content("Updates"),
            E.p().content("An updated version of Enfugue is available. See below for details regarding new features and fixes."),
            E.p().content(
                E.span().content("If you are managing your own installation, simply executing "),
                E.createElement("code").content("pip install enfugue --update"),
                E.span().content(" will fetch the latest details. If you are using an easy installation, you will need to download the updated package. See below for any additional steps required.")
            ),
            E.p().class("center").content(
                E.a().class("button").href(this.constructor.releasesLink).target("_blank").content("Download Now")
            )
        );
        return node;
    }
}

/**
 * This class displays an individual update/announcement
 */
class UpdateAnnouncementView extends View {
    constructor(config, update) {
        super(config);
        this.update = update;
    }

    /**
     * On build, append details of version
     */
    async build() {
        let node = await super.build();
        node.content(
            E.h3().content(`Version ${this.update.version}`),
            E.p().addClass("note").content(`Released ${this.update.release}`),
            E.p().content(this.update.description)
        );
        return node;
    }
}

/**
 * This class manages announcements from the server.
 */
class AnnouncementsController extends Controller {
    /**
     * @var int The default announcements window width
     */
    static announcementsWindowWidth = 800;

    /**
     * @var int The default announcements window height
     */
    static announcementsWindowHeight = 1000;
    
    /**
     * Checks for announcements and shows them if needed
     */
    async checkShowAnnouncements() {
        let activeAnnouncements = await this.model.get("/announcements");
        if (!isEmpty(activeAnnouncements)) {
            let initializeAnnouncement = activeAnnouncements.filter(
                    (announcement) => announcement.type === "initialize"
                ),
                downloadAnnouncements = activeAnnouncements.filter(
                    (announcement) => announcement.type === "download"
                ),
                optionalDownloadAnnouncements = activeAnnouncements.filter(
                    (announcement) => announcement.type === "optional-download"
                ),
                updateAnnouncements = activeAnnouncements.filter(
                    (announcement) => announcement.type === "update"
                ),
                acknowledgeButton;

            let announcementsView = new ParentView(this.config);
            await announcementsView.addClass("announcements-view");
            if (!isEmpty(initializeAnnouncement)) {
                let initializationAnnouncementView = await announcementsView.addChild(
                        InitializationAnnouncementView, 
                        initializeAnnouncement[0].directories
                    ),
                    directoryForm = initializationAnnouncementView.formView;

                directoryForm.onChange(() => {
                    acknowledgeButton.disable();
                    directoryForm.setError("You must submit directory changes before continuing.");
                });

                directoryForm.onSubmit(async (newDirectories) => {
                    try {
                        await this.model.post("installation", null, null, {"directories": newDirectories});
                    } catch(e) {
                        let errorMessage = `${e}`;
                        if (!isEmpty(e.status) && e.status == 400) {
                            // one or more directory doesn't exist, offer to create
                            let hasMultiple = `${e.detail}`.indexOf(",") !== -1,
                                itThem = hasMultiple ? "them" : "it";
                            if (await this.application.yesNo(`${e.detail}<br/><br/>Create ${itThem}?`)) {
                                // pass create, raise details
                                await this.model.post("installation", null, null, {"directories": newDirectories, "create": true});
                            } else {
                                directoryForm.setError(e);
                                directoryForm.enable();
                                return;
                            }
                        } else {
                            throw e;
                        }
                    }
                    this.notify("info", "Success", "Directories successfully set.");
                    directoryForm.clearError();
                    directoryForm.enable();
                    acknowledgeButton.enable();
                });
            }
            if (!isEmpty(downloadAnnouncements)) {
                await announcementsView.addChild(DownloadAnnouncementHeaderView);
                let tableView = await announcementsView.addChild(DownloadTableView, downloadAnnouncements);
                tableView.addButton("Download", "fa-solid fa-download", async (row) => {
                    let tableRows = (await tableView.getNode()).find("tbody").findAll("tr");
                    for (let tableRow of tableRows) {
                        if (tableRow.find("td").getText() == row.model) {
                            tableRow.find("button").disabled(true).addClass("disabled");
                        }
                    }
                    this.download("checkpoint", row.url, row.model);
                });
            }
            if (!isEmpty(optionalDownloadAnnouncements)) {
                await announcementsView.addChild(OptionalDownloadAnnouncementHeaderView);
                let tableView = await announcementsView.addChild(DownloadTableView, optionalDownloadAnnouncements);
                tableView.addButton("Download", "fa-solid fa-download", async (row) => {
                    let tableRows = (await tableView.getNode()).find("tbody").findAll("tr");
                    for (let tableRow of tableRows) {
                        if (tableRow.find("td").getText() == row.model) {
                            tableRow.find("button").disabled(true).addClass("disabled");
                        }
                    }
                    this.download("checkpoint", row.url, row.model);
                });
            }
            if (!isEmpty(updateAnnouncements)) {
                await announcementsView.addChild(UpdateAnnouncementHeaderView);
                for (let update of updateAnnouncements) {
                    await announcementsView.addChild(UpdateAnnouncementView, update);
                }
            }
            if (!announcementsView.isEmpty()) {
                acknowledgeButton = await announcementsView.addChild(AcknowledgeButtonInputView);
                let announcementsWindow = await this.spawnWindow(
                    "Announcements",
                    announcementsView,
                    this.constructor.announcementsWindowWidth,
                    this.constructor.announcementsWindowHeight
                );
                acknowledgeButton.onChange(() => {
                    announcementsWindow.remove();
                });
                announcementsWindow.onClose(async () => {
                    await this.model.post("/announcements/snooze");
                });
                if (!isEmpty(initializeAnnouncement)) {
                    // Disable hiding
                    let buttons = (await announcementsWindow.getNode()).findAll("enfugue-node-button");
                    for (let button of buttons) {
                        button.hide();
                    }
                }
            }
        }
    }

    /**
     * on initialize, check and show announcements. Set a timer to do this once a day.
     */
    async initialize() {
        this.checkShowAnnouncements();
        setInterval(() => this.checkShowAnnouncements(), 1000 * 60 * 60 * 24);
    }
}

export { AnnouncementsController };
