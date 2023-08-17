/** @module controller/common/logs */
import { isEmpty, waitFor, deepClone } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { TableView } from "../../view/table.mjs";
import { View } from "../../view/base.mjs";

const E = new ElementBuilder();

/**
 * This class shows a very small amount of logs to the user; this is mostly
 * just so the user feels good that it's doing something, if thy want to view
 * the rest of the logs they should click the 'Show More' button
 */
class LogGlanceView extends View {
    /**
     * @var string custom tag name
     */
    static tagName = "enfugue-log-glance-view";

    /**
     * @var int The maximum number of logs to show.
     */
    static maximumLogs = 5;

    /**
     * On construct, set time and hide ourselves
     */
    constructor(config) {
        super(config);
        this.start = (new Date()).getTime();
        this.hide();
    }

    /**
     * Show more details. Callback is set by the controller.
     */
    async showMore() {
        if (!isEmpty(this.onShowMore)) {
            await this.onShowMore();
        }
    }

    /**
     * Set log data after build.
     */
    async setData(logs) {
        if (this.node !== undefined) {
            let logsToShow = logs.filter((log) => {
                let logTime = (new Date(log.timestamp)).getTime();
                return logTime > this.start;
            });
            if (logsToShow.length > 0) {
                let currentTime = (new Date()).getTime(),
                    logText = logsToShow.slice(0, this.constructor.maximumLogs).map((log) => log.content).join("\n");
                this.show();
                this.node.find(".logs").content(logText);
            }
        }
    }

    /**
     * On build, add DOM elements and buttons
     */
    async build() {
        let node = await super.build();
        node.append(
            E.div().class("log-header").content(
                E.h2().content("Most Recent Logs"),
                E.a().href("#").content("Show More").on("click", (e) => { e.preventDefault(); e.stopPropagation(); this.showMore(); })
            ),
            E.div().class("logs")
        );
        return node;
    }
};

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

    /**
     * @var object format columns
     */
    static columnFormatters = {
        "timestamp": (ts) => ts.replace("T", " ").split(".")[0]
    };
}

/**
 * The LogsController allows tailing the log for increased visibility
 */
class LogsController extends Controller {
    /**
     * @var int The width of the logs window
     */
    static logsWindowWidth = 600;

    /**
     * @var int The height of the logs window
     */
    static logsWindowHeight = 700;

    /**
     * @var int Maximum logs to show at one time in the detail window
     */
    static maximumDetailLogs = 100;

    /**
     * @var int Maximum logs to show at once in the glance window
     */
    static maximumGlanceLogs = 5;

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
        
        if (isEmpty(this.logs)) {
            this.logs = result;
        } else {
            this.logs = result.concat(this.logs);
        }
        this.logs = this.logs.slice(0, this.constructor.maximumDetailLogs);
        return this.logs;
    }

    /**
     * Gets the System Logs View
     */
    async getLogsTable() {
        if (isEmpty(this.logsTable)) {
            await waitFor(() => this.logs !== null && this.logs !== undefined);
            this.logsTable = new LogTableView(this.config, this.logs);
        }
        return this.logsTable;
    };

    /**
     * Starts the interval that tails the log
     */
    async startLogTailer() {
        this.timer = setInterval(async () => {
            let newLogs = await this.getLogs();
            if (!isEmpty(this.logsTable)) {
                this.logsTable.setData(newLogs, false);
            }
            if (!isEmpty(this.glanceView)) {
                this.glanceView.setData(newLogs);
            }
        }, this.constructor.logTailInterval);
    }

    /**
     * Builds the manager if not yet built.
     */
    async showLogDetails() {
        if (isEmpty(this.logDetails)) {
            this.logWindow = await this.spawnWindow(
                "Engine Logs",
                await this.getLogsTable(),
                this.constructor.logsWindowWidth,
                this.constructor.logsWindowHeight
            );
            this.logWindow.onClose(() => { this.logWindow = null; });
        } else {
            this.logWindow.focus();
        }
    }

    /**
     * On init, add glance view
     */
    async initialize() {
        this.glanceView = new LogGlanceView(this.config);
        this.glanceView.onShowMore = () => this.showLogDetails();
        this.application.container.appendChild(await this.glanceView.render());
        this.startLogTailer();
    }
};

export { LogsController };
