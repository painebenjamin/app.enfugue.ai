/** @module controller/common/logs */
import { isEmpty, waitFor, deepClone } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { ButtonInputView } from "../../forms/input/misc.mjs";
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
    static maximumLogs = 10;

    /**
     * On construct, set time and hide ourselves
     */
    constructor(config) {
        super(config);
        this.start = (new Date()).getTime();
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
            E.div().class("logs").content("Welcome to ENFUGUE! When the diffusion engine logs to file, the most recent lines will appear here.")
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
 * This view includes the logs table and a button to pause
 */
class LogsView extends View {
    /**
     * @var string The tag name for the log view
     */
    static tagName = "enfugue-logs-view";

    /**
     * On construct, build logs and button
     */
    constructor(config, data) {
        super(config);
        this.paused = false;
        this.logsTable = new LogTableView(config, data);
        this.pauseLogs = new ButtonInputView(config, "pause", {"value": "Pause Logs"});
        this.pauseLogs.onChange(() => {
            this.paused = !this.paused;
            if (this.paused) {
                this.pauseLogs.setValue("Unpause Logs", false);
            } else {
                this.pauseLogs.setValue("Pause Logs", false);
            }
        });
    }

    /**
     * Sets the data in the logs table (if not paused)
     */
    setData(newData) {
        if (!this.paused) {
            this.logsTable.setData(newData, false);
        }
    }

    /**
     * On build, get nodes
     */
    async build() {
        let node = await super.build();
        node.append(
            await this.pauseLogs.getNode(),
            await this.logsTable.getNode()
        );
        return node;
    }
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
    async getLogsView() {
        if (isEmpty(this.logsView)) {
            await waitFor(() => this.logs !== null && this.logs !== undefined);
            this.logsView = new LogsView(this.config, this.logs);
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
                this.logsView.setData(newLogs);
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
        if (isEmpty(this.logWindow)) {
            this.logWindow = await this.spawnWindow(
                "Engine Logs",
                await this.getLogsView(),
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
        this.application.sidebar.addChild(this.glanceView);
        this.startLogTailer();
    }
};

export {
    LogsController as SidebarController
};
