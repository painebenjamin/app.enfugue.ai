/** @module controller/models/04-merge */
import { isEmpty, humanSize } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { MergeModelsFormView } from "../../forms/enfugue/models.mjs";

/**
 * Shows the 'merge models' form
 */
class MergeModelsController extends MenuController {
    /**
     * @var int width of the merge window
     */
    static modelMergeWindowWidth = 400;

    /**
     * @var int height of the merge window
     */
    static modelMergeWindowHeight = 600;

    /**
     * @var string The text in the UI
     */
    static menuName = "Merge Models";
    
    /**
     * @var string The class of the icon in the UI
     */
    static menuIcon = "fa-solid fa-code-merge";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "e";

    /**
     * Show the new model form when clicked
     */
    async onClick() {
        this.showModelMerger();
    }

    /**
     * Shows the merger.
     * Creates if not yet done.
     */
    async showModelMerger() {
        if (!isEmpty(this.modelMergerWindow)) {
            this.modelMergeWindow.focus();
            return;
        }
        let modelMergeForm = new MergeModelsFormView(this.config);
        modelMergeForm.onSubmit(async (values) => {
            modelMergeForm.clearError();
            try {
                let result = await this.model.post("/model-merge", null, null, values),
                    resultMessage = `Wrote ${humanSize(result.size)} to ${result.path}`;
                this.notify("info", "Models Successfully Merged", resultMessage);
                this.modelMergeWindow.remove();
                this.modelMergeWindow = null;
            } catch(e) {
                modelMergeForm.setError(e);
                modelMergeForm.enable();
            }
        });
        this.modelMergeWindow = await this.spawnWindow(
            "Merge Models",
            modelMergeForm,
            this.constructor.modelMergeWindowWidth,
            this.constructor.modelMergeWindowHeight
        );
    }
}

export { MergeModelsController as MenuController }
