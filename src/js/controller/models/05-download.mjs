/** @module controller/models/04-download */
import { isEmpty, humanSize } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { DownloadModelsFormView } from "../../forms/enfugue/models.mjs";

/**
 * Shows the 'download models' form
 */
class DownloadModelsController extends MenuController {
    /**
     * @var int width of the download window
     */
    static modelDownloadWindowWidth = 400;

    /**
     * @var int height of the download window
     */
    static modelDownloadWindowHeight = 275;

    /**
     * @var string The text in the UI
     */
    static menuName = "Download Models";
    
    /**
     * @var string The class of the icon in the UI
     */
    static menuIcon = "fa-solid fa-cloud-arrow-down";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "d";

    /**
     * Show the new model form when clicked
     */
    async onClick() {
        this.showModelDownloader();
    }

    /**
     * Shows the downloadr.
     * Creates if not yet done.
     */
    async showModelDownloader() {
        if (!isEmpty(this.modelDownloaderWindow)) {
            this.modelDownloadWindow.focus();
            return;
        }
        let modelDownloadForm = new DownloadModelsFormView(this.config);
        modelDownloadForm.onSubmit(async (values) => {
            modelDownloadForm.clearError();
            try {
                let result = await this.model.post("/download", null, null, values);
                this.notify("info", "Download Queued", "The download was successfully queued, it will begin shortly.");
                this.modelDownloadWindow.remove();
                this.modelDownloadWindow = null;
            } catch(e) {
                modelDownloadForm.setError(e);
                modelDownloadForm.enable();
            }
        });
        this.modelDownloadWindow = await this.spawnWindow(
            "Download Model",
            modelDownloadForm,
            this.constructor.modelDownloadWindowWidth,
            this.constructor.modelDownloadWindowHeight
        );
    }
}

export { DownloadModelsController as MenuController };
