/** @module controller/extras/02-stable-video */
import { isEmpty, isEquivalent, sleep } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { View } from "../../view/base.mjs";
import { MenuController } from "../menu.mjs";
import { StableVideoDiffusionFormView } from "../../forms/enfugue/animation.mjs";

const E = new ElementBuilder({});

/**
 * Shows the Stable Video Diffusion form and bind events.
 */
class StableVideoController extends MenuController {
    /**
     * @var int width of the input window
     */
    static stableVideoWindowWidth = 400;

    /**
     * @var int height of the input window
     */
    static stableVideoWindowHeight = 1030;

    /**
     * @var string The text in the UI
     */
    static menuName = "Stable Video (SVD)";
    
    /**
     * @var string The class of the icon in the UI
     */
    static menuIcon = "fa-solid fa-video";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "v";

    /**
     * Show the new model form when clicked
     */
    async onClick() {
        this.showStableVideo();
    }

    /**
     * Shows the form.
     * Creates if not yet done.
     */
    async showStableVideo() {
        if (!isEmpty(this.stableVideoWindow)) {
            this.stableVideoWindow.focus();
            return;
        }
        let stableVideoForm = new StableVideoDiffusionFormView(this.config);
        stableVideoForm.onSubmit(async (values) => {
            stableVideoForm.clearError();
            try {
                let result = await this.model.post("/invoke/svd", null, null, values);
                if (isEmpty(result.uuid)) {
                    throw "Response did not contain a result.";
                }
                this.engine.enableStop();
                this.engine.startSample = true;
                this.engine.canvasInvocation(result.uuid, true);
                this.notify("info", "Success", "Invocation queued, it will begin shortly.");
                stableVideoForm.enable();
            } catch(e) {
                stableVideoForm.setError(e);
                stableVideoForm.enable();
            }
        });
        this.stableVideoWindow = await this.spawnWindow(
            "Stable Video Diffusion",
            stableVideoForm,
            this.constructor.stableVideoWindowWidth,
            this.constructor.stableVideoWindowHeight
        );
        this.stableVideoWindow.onClose(() => {
            this.stableVideoWindow = null;
        });
    }
}

export { StableVideoController as MenuController };
