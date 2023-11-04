/** @module controller/file/05-results */
import { MenuController } from "../menu.mjs";
import { ModelTableView } from "../../view/table.mjs";
import { ImageView } from "../../view/image.mjs";
import { isEmpty, humanDuration, sleep } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";

const E = new ElementBuilder({
    "invocationOutputs": "enfugue-invocation-outputs",
    "invocationOutput": "enfugue-invocation-output"
});

/**
 * Extends the ModelTableView to add formatters for columns
 */
class InvocationTableView extends ModelTableView {
    /**
     * @var string The classname for the node
     */
    static className = "invocation-history";

    /**
     * @var string The searchable columns
     */
    static searchFields = ["plan"];

    /**
     * @var array<object> The configuration for the buttons on this row.
     */
    static buttons = [
        {
            "icon": "fa-solid fa-trash",
            "label": "Delete",
            "click": async function(datum) {
                await InvocationTableView.deleteInvocation(datum.id); // Set at init
                await sleep(100); // Wait a tick
                await this.parent.requery();
            }
        }
    ];

    /**
     * @var array<object> The initial sort for invocations - most recent first.
     */
    static sortGroups = [
        {
            "column": "started",
            "direction": "desc"
        }
    ];

    /**
     * @var object The columns and labels for values from kwargs to print out.
     */
    static printKwargs = {
        "prompt": "Prompt",
        "negative_prompt": "Negative Prompt",
        "width": "Width",
        "height": "Height",
        "image_width": "Engine Width",
        "image_height": "Engine Height",
        "model": "Checkpoint",
        "inversion": "Textual Inversion(s)",
        "lora": "LoRA(s)"
    };

    /**
     * @var object<callable> How to format individual columns.
     */
    static columnFormatters = {
        "duration": (value) => humanDuration(parseFloat(value), true, true),
        "plan": (plan) => {
            return JSON.stringify(plan);
        },
        "prompt": (_, datum) => datum.plan.prompt,
        "seed": (_, datum) => `${datum.plan.seed}`,
        "outputs": async function(outputCount, datum) {
            if (outputCount > 0) {
                let outputContainer = E.invocationOutputs();
                if (!isEmpty(datum.plan.animation_frames) && datum.plan.animation_frames > 0) {
                    let thumbnailVideoSource = `/api/invocation/animation/thumbnails/${datum.id}.mp4`,
                        imageContainer = E.invocationOutput()
                            .content(
                                E.video()
                                    .content(E.source().src(thumbnailVideoSource))
                                    .autoplay(true)
                                    .muted(true)
                                    .loop(true)
                            );

                     outputContainer.append(imageContainer);
                } else {
                    for (let i = 0; i < outputCount; i++) {
                        let imageName = `${datum.id}_${i}.png`,
                            imageSource = `/api/invocation/images/${imageName}`,
                            thumbnailSource = `/api/invocation/thumbnails/${imageName}`,
                            imageView = new ImageView(this.config, thumbnailSource, false),
                            imageContainer = E.invocationOutput()
                                .content(await imageView.getNode())
                                .on("click", async () => {
                                    InvocationTableView.setCurrentInvocationImage(imageSource); // Set at init
                                });
                        outputContainer.append(imageContainer);
                    }
                }
                return outputContainer;
            } else if(!isEmpty(datum.error)) {
                return `Error: ${datum.error}`;
            } else {
                return "None";
            }
        }
    };

    /**
     * @var object The columns for this table
     */
    static columns = {
        "started": "Started",
        "duration": "Duration",
        "seed": "Seed",
        "prompt": "Prompt",
        "plan": "Parameters",
        "outputs": "Output"
    };
};
 
/**
 * The history controller allows a user to see their past invocations.
 */
class ResultsController extends MenuController {
    /**
     * @var int The initial width of the history window
     */
    static historyTableWidth = 1000;
    
    /**
     * @var int The initial height of the history window
     */
    static historyTableHeight = 500;

    /**
     * @var string The text in the menu
     */
    static menuName = "Results";

    /**
     * @var string The classes of the <i> element in the menu
     */
    static menuIcon = "fa-solid fa-images";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "r";

    /**
     * On initialization, register callbacks with table view.
     */
    async initialize() {
        await super.initialize();
        InvocationTableView.deleteInvocation = (id) => { this.model.delete(`/invocation/${id}`); };
        InvocationTableView.setCurrentInvocationImage = (image) => this.application.images.setCurrentInvocationImage(image);

    }

    /**
     * Creates (spawns) the history window
     */
    async createResultsWindow() {
        return await this.spawnWindow(
            "Results",
            new InvocationTableView(this.config, this.model.DiffusionInvocation),
            this.constructor.historyTableWidth,
            this.constructor.historyTableHeight
        );
    }

    /**
     * When clicked, open the history window.
     * Refresh it if it's already open.
     */
    async onClick() {
        if (!isEmpty(this.historyWindow)) {
            this.historyWindow.focus();
        } else {
            this.historyWindow = await this.createResultsWindow();
            this.historyWindow.onClose(() => { delete this.historyWindow; });
        }
    }
}

export { ResultsController as MenuController };
