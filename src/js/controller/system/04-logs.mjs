/** @module controller/system/04-logs */
import { isEmpty, deepClone } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { FormView } from "../../view/forms/base.mjs";
import { TableView } from "../../view/table.mjs";
import { ParentView } from "../../view/base.mjs";
import { StringInputView, ListMultiInputView } from "../../view/forms/input.mjs";

/**
 * This class lets you select from lgo levels
 * TODO: Expose this
 */
class LogLevelSelectInputView extends ListMultiInputView {
    /**
     * @var array The log level options
     */
    static defaultOptions = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"];
}

/**
 * This class displays logs in a nice table
 */
class LogTableView extends TableView {
    /**
     * @var string Custom css classname
     */
    static className = "log-table-view";

    /**
     * @var bool Disable sorting
     */
    static canSort = false;

    /**
     * @var bool Disable default sort
     */
    static applyDefaultSort = false;

    /**
     * @var object Columns and labels
     */
    static columns = {
        "timestamp": "Timestamp",
        "logger": "Logger",
        "level": "Level",
        "content": "Content"
    };
}

/**
 * This class assembles the filter options for logs
 */
class LogFilterFormView extends FormView {
    /**
     * @var bool Disable submit button
     */
    static autoSubmit = true;

    /**
     * @var objects filter fieldsets
     */
    static fieldSets = {
        "Filters": {
            "level": {
                "label": "Log Levels",
                "class": LogLevelSelectInputView,
                "config": {
                    "value": deepClone(LogLevelSelectInputView.defaultOptions)
                }
            },
            "search": {
                "label": "Search",
                "class": StringInputView
            }
        }
    };
};

/**
 * The SystemLogsController allows tailing the log for increased visibility
 */
class SystemLogsController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "Engine Logs";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-clipboard-list";

    /**
     * @var int The width of the logs window
     */
    static logsWindowWidth = 600;

    /**
     * @var int The height of the logs window
     */
    static logsWindowHeight = 700;

    /**
     * @var int Maximum logs to show at one time
     */
    static maximumLogs = 100;

    /**
     * @var int Log tail interval in MS
     */
    static logTailInterval = 5000;

    /**
     * Gets the logs from the API
     */
    async getLogs() {
        let params = {};
        if (!isEmpty(this.lastLog)) {
            let logTimestamp = `${this.lastLog.getFullYear()}-${(this.lastLog.getMonth()+1).toString().padStart(2, '0')}-${this.lastLog.getDate().toString().padStart(2, '0')} ${this.lastLog.getHours().toString().padStart(2, '0')}:${this.lastLog.getMinutes().toString().padStart(2, '0')}:${this.lastLog.getSeconds().toString().padStart(2, '0')}`;
            params["since"] = logTimestamp;
        }
        if (!isEmpty(this.levels)) {
            params["level"] = this.levels;
        }
        if (!isEmpty(this.search)) {
            params["search"] = this.search;
        }
        let result = await this.model.get("/logs", null, params);
        this.lastLog = new Date();
        // Only ever show 100 in the DOM to not make it slow down
        if (isEmpty(this.logs)) {
            this.logs = result;
        } else {
            this.logs = result.concat(this.logs);
        }
        this.logs = this.logs.slice(0, this.constructor.maximumLogs);
        return this.logs;
    }

    /**
     * Gets the System Logs View
     */
    async getLogsView() {
        if (isEmpty(this.logsView)) {
            let currentLogs = await this.getLogs();
            this.logsView = new ParentView(this.config);
            //this.filterOptions = await this.logsView.addChild(LogFilterFormView); TODO
            this.logsTable = await this.logsView.addChild(LogTableView, currentLogs);
        }
        return this.logsView;
    };

    /**
     * Starts the interval that tails the log
     */
    async startLogTailer() {
        this.timer = setInterval(async () => {
            let newLogs = await this.getLogs();

            if (!isEmpty(this.logsView)) {

                this.logsTable.setData(newLogs, false);
            }
        }, this.constructor.logTailInterval);
    }

    /**
     * Stops the interval that tails the log
     */
    async stopLogTailer() {
        clearInterval(this.timer);
        this.logTailer = null;
    }

    /**
     * Builds the manager if not yet built.
     */
    async showLogTailer() {
        if (isEmpty(this.logTailer)) {
            this.logTailer = await this.spawnWindow(
                "Engine Logs",
                await this.getLogsView(),
                this.constructor.logsWindowWidth,
                this.constructor.logsWindowHeight
            );
            this.startLogTailer();
            this.logTailer.onClose(() => { this.stopLogTailer(); });
        } else {
            this.logTailer.focus();
        }
    }

    /**
     * When clicked, show logs window.
     */
    async onClick() {
        this.showLogTailer();
    }
};

export { SystemLogsController as MenuController };
