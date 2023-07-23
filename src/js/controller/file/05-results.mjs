/** @module controller/file/history */
import { MenuController } from "../menu.mjs";
import { ModelTableView } from "../../view/table.mjs";
import { ImageView, ImageInspectorView } from "../../view/image.mjs";
import { isEmpty, humanDuration, sleep } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";

const imageWidthPadding = 2,
      imageHeightPadding = 100,
      E = new ElementBuilder({
        "invocationOutputs": "enfugue-invocation-outputs",
        "invocationOutput": "enfugue-invocation-output"
      });

/**
 * Extends the ModelTableView to add formatters for columns
 */
class InvocationTableView extends ModelTableView {
    /**
     * @var int The width of the image inspector view when it's made
     */
    static imageInspectorWidth = 550;
    
    /**
     * @var int The height of the image inspector view when it's made
     */
    static imageInspectorHeight = 550;

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
        "duration": (value) => humanDuration(parseFloat(value)),
        "plan": (plan) => {
            return JSON.stringify(plan);
        },
        "outputs": async function(outputCount, datum) {
            if (outputCount > 0) {
                let outputContainer = E.invocationOutputs();
                for (let i = 0; i < outputCount; i++) {
                    let imageName = `${datum.id}_${i}.png`,
                        imageSource = `/api/invocation/images/${imageName}`,
                        thumbnailSource = `/api/invocation/thumbnails/${imageName}`,
                        imageView = new ImageView(this.config, thumbnailSource),
                        imageInspector = new ImageInspectorView(
                            this.config,
                            imageSource,
                            imageName,
                            InvocationTableView.imageInspectorWidth,
                            InvocationTableView.imageInspectorHeight
                        ),
                        imageContainer = E.invocationOutput()
                            .content(await imageView.getNode())
                            .on("click", async () => {
                                let invocationWindow = await InvocationTableView.spawnWindow( // Set at init
                                    `${datum.id} sample ${i+1}`,
                                    imageInspector,
                                    InvocationTableView.imageInspectorWidth + 2,
                                    InvocationTableView.imageInspectorHeight + 32
                                );
                                invocationWindow.onResize((arg) => {
                                    imageInspector.setDimension(
                                        invocationWindow.visibleWidth - 23,
                                        invocationWindow.visibleHeight - 53
                                    );
                                });
                            });
                    outputContainer.append(imageContainer);
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
    static historyTableWidth = 800;
    
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
     * On initialization, register callbacks with table view.
     */
    async initialize() {
        await super.initialize();
        InvocationTableView.spawnWindow = (name, content, width, height) => this.spawnWindow(name, content, width, height);
        InvocationTableView.deleteInvocation = (id) => { this.model.delete(`/invocation/${id}`); };
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
