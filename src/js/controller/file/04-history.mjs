/** @module controller/file/04-history */
import { isEmpty, humanDuration, sleep } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { ParentView } from "../../view/base.mjs";
import { TableView } from "../../view/table.mjs";
import { StringInputView } from "../../forms/input.mjs";
import { ImageView, ImageInspectorView } from "../../view/image.mjs";
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
class HistoryTableView extends TableView {
    /**
     * @var array<object> The configuration for the buttons on this row.
     */
    static buttons = [
        {
            "icon": "fa-solid fa-repeat",
            "label": "Reload",
            "click": async function(datum) {
                await HistoryTableView.reloadHistory(datum); // Set at init
            }
        },
        {
            "icon": "fa-solid fa-trash",
            "label": "Delete",
            "click": async function(datum) {
                await HistoryTableView.deleteHistory(datum); // Set at init
            }
        }
    ];

    /**
     * @var array<object> The initial sort for history
     */
    static sort = [["timestamp", true]];

    /**
     * @var object Column formatters, we use this for timestamp sort to work correctly
     */
    static columnFormatters = {
        "timestamp": (value) => (new Date(value)).toLocaleString()
    };

    /**
     * @var object The columns and labels
     */
    static columns = {
        "timestamp": "Timestamp",
        "summary": "Summary"
    };

    /**
     * @var string Custom classname for CSS
     */
    static className = "history-table-view";
};

/**
 * The history controller allows a user to see their past invocations.
 */
class HistoryController extends MenuController {
    /**
     * @var int The initial width of the history window
     */
    static historyTableWidth = 800;
    
    /**
     * @var int The initial height of the history window
     */
    static historyTableHeight = 500;

    /**
     * @var int The maximum height of the preview
     */
    static historyPreviewHeight = 80;

    /**
     * @var string The text in the menu
     */
    static menuName = "History";

    /**
     * @var string The classes of the <i> element in the menu
     */
    static menuIcon = "fa-solid fa-clock-rotate-left";

    /**
     * @var object Config to pass into the string input
     */
    static searchHistoryConfig = {
        "placeholder": "Start typing to search…"
    };

    /**
     * @var int The number of milliseconds to wait after input before searching
     */
    static searchHistoryDebounceDelay = 250;

    /**
     * On initialization, register callbacks with table view.
     */
    async initialize() {
        await super.initialize();
        HistoryTableView.reloadHistory = (datum) => {
            this.application.setState(datum.state, true);
            this.application.history.deleteByID(datum.id);
            let timestamp = (new Date(datum.timestamp)).toLocaleString();
            this.notify("info", "Session Reloaded", `The session from ${timestamp} has been successfully restored.`);
            setTimeout(() => this.resetHistory(), 150);
        };
        HistoryTableView.deleteHistory = (datum) => {
            this.application.history.deleteByID(datum.id);
            setTimeout(() => this.resetHistory(), 150);
        };
        this.searchText = "";
    }

    /**
     * Re-queries and resets the history table.
     */
    resetHistory() {
        return new Promise((resolve, reject) => {
            this.getHistory().then((newData) => {
                this.historyTable.setData(newData, false);
                resolve();
            }).catch(reject);
        });
    }
    
    /**
     * Gets history from the browser's database
     */
    async getHistory() {
        let historyItems;
        
        if (isEmpty(this.searchText)) {
            historyItems = await this.application.history.getHistoryItems();
        } else {
            historyItems = await this.application.history.searchHistoryItems(this.searchText);
        }

        return historyItems.map((item) => {
            let summaryParts = {};
            if (!isEmpty(item.state.model)) {
                summaryParts["Model"] = item.state.model.model;
            }
            if (!isEmpty(item.state.canvas)) {
                summaryParts["Dimensions"] = `${item.state.canvas.width}px × ${item.state.canvas.height}px`;
            }
            if (!isEmpty(item.state.prompts)) {
                summaryParts["Prompt"] = item.state.prompts.prompt;
                if (!isEmpty(item.state.prompts.negativePrompt)) {
                    summaryParts["Negative Prompt"] = item.state.prompts.negativePrompt;
                }
            }
            if (!isEmpty(item.state.images)) {
                let nodeNumber = 1;
                for (let node of item.state.images) {
                    let nodeSummaryParts = {};
                    switch (node.classname) {
                        case "ImageEditorImageNodeView":
                            nodeSummaryParts["Type"] = "Image";
                            let features = [];
                            if (node.inpaint) {
                                features.push("Inpainting");
                            }
                            if (node.infer) {
                                features.push("Inference");
                            }
                            if (node.control) {
                                features.push("ControlNet");
                            }
                            if (node.removeBackground) {
                                features.push("Background Removal");
                            }
                            if (features.length > 0) {
                                nodeSummaryParts["Features"] = features.join(", ");
                            }
                            break;
                        case "ImageEditorPromptNodeView":
                            nodeSummaryParts["Type"] = "Prompt";
                            break;
                        case "ImageEditorScribbleNodeView":
                            nodeSummaryParts["Type"] = "Scribble";
                            break;
                    }
                    nodeSummaryParts["Dimensions"] = `${node.w}px × ${node.h}px`;
                    nodeSummaryParts["Position"] = `(${node.x}px, ${node.y}px)`;
                    if (!isEmpty(node.prompt)) {
                        nodeSummaryParts["Prompt"] = node.prompt;
                    }
                    let nodeSummaryString = Object.getOwnPropertyNames(nodeSummaryParts).map((key) => {
                        return `<u>${key}</u>: ${nodeSummaryParts[key]}`;
                    }).join(", ");
                    summaryParts[`Node ${nodeNumber++}`] = `<span style='font-size: 0.8em'>${nodeSummaryString}</span>`;
                }
            }
            
            let summaryString = Object.getOwnPropertyNames(summaryParts).map((key) => {
                return `<strong>${key}</strong>: ${summaryParts[key]}`;
            }).join(", ");

            return {
                "id": item.id,
                "state": item.state,
                "timestamp": item.timestamp,
                "summary": summaryString
            };
        });
    }

    /**
     * Creates (spawns) the history window
     */
    async createHistoryWindow() {
        this.searchHistory = new StringInputView(this.config, "search", this.constructor.searchHistoryConfig);
        this.searchHistory.onInput((searchValue) => {
            clearTimeout(this.searchTimer);
            this.searchTimer = setTimeout(() => {
                this.searchText = searchValue;
                this.resetHistory();
            }, this.constructor.searchHistoryDebounceDelay);
        });
        this.historyTable = new HistoryTableView(this.config, await this.getHistory());
        
        let container = new ParentView(this.config);
        container.addClass("history-view");
        await container.addChild(this.searchHistory);
        await container.addChild(this.historyTable);
        return await this.spawnWindow(
            "History",
            container,
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
            this.historyWindow = await this.createHistoryWindow();
            this.historyWindow.onClose(() => { delete this.historyWindow; });
        }
    }
}

export { HistoryController as MenuController };
