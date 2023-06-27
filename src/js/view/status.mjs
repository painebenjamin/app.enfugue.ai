/** @module view/status */
import { isEmpty, humanSize, humanDuration } from "../base/helpers.mjs";
import { View } from "./base.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { ColorScale, getTextColorForBackground } from "../graphics/colors.mjs";

const E = new ElementBuilder({
    "statusIcon": "enfugue-status-icon",
    "statusVersion": "enfugue-status-version",
    "statusUptime": "enfugue-status-uptime",
    "statusGpuName": "enfugue-status-gpu-name",
    "statusGpuTemp": "enfugue-status-gpu-temp",
    "statusGpuLoad": "enfugue-status-gpu-load",
    "statusGpuMemory": "enfugue-status-gpu-memory"
});

/**
 * The color scale goes G->Y->R
 */
const statusColorScale = new ColorScale([
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]
]);

/**
 * The main view takes the status and renders details to the header.
 */
class StatusView extends View {
    /**
     * @var string The tag name of the view element
     */
    static tagName = "enfugue-status";

    /**
     * @param object $config The main config object.
     * @param object $status The status object from the API.
     */
    constructor(config, status) {
        super(config);
        this.status = status;
        this.statusClickCallbacks = [];
    }

    /**
     * Adds a callback to perform when the status button is clicked
     */
    onStatusClicked(callback){
        this.statusClickCallbacks.push(callback);
    }

    /**
     * The function fired when the status button is clicked.
     */
    async statusClicked() {
        if (isEmpty(this.status) || this.label === "error" || this.label === "idle") {
            return;
        }
        for (let callback of this.statusClickCallbacks) {
            await callback();
        }
    }

    /**
     * @return string The version string from the status object
     */
    get version() {
        return this.status.version;
    }

    /**
     * @return string The human-formatted uptime
     */
    get uptime() {
        return humanDuration(this.status.uptime);
    }

    /**
     * @return string The label of the status itself
     */
    get label() {
        return this.status.status;
    }

    /**
     * @return int The GPU load in integer percentage
     */
    get gpuLoad() {
        return Math.ceil(this.status.gpu.load * 100);
    }

    /**
     * @return float The GPU memory total in GB
     */
    get gpuMemoryTotal() {
        return this.status.gpu.memory.total;
    }

    /**
     * @return float The GPU memory free (unused) in GB
     */
    get gpuMemoryFree() {
        return this.status.gpu.memory.free;
    }

    /**
     * @return float The GPU memory used in GB
     */
    get gpuMemoryUsed() {
        return this.status.gpu.memory.used;
    }

    /**
     * @return int The GPU memory load in integer percentage
     */
    get gpuMemoryUtil() {
        return Math.ceil((this.gpuMemoryUsed / this.gpuMemoryTotal) * 100);
    }

    /**
     * @return string A description string of the current memory
     */
    get gpuMemoryUsage() {
        return `${humanSize(this.gpuMemoryUsed*1e6)} / ${humanSize(this.gpuMemoryTotal*1e6)}`;
    }

    /**
     * @return string The name of the GPU
     */
    get gpuName() {
        return this.status.gpu.name;
    }

    /**
     * @return int The temperature of the GPU in celsius
     */
    get gpuTemp() {
        return this.status.gpu.temp;
    }

    /**
     * Updates the status. Will re-render node if needed.
     * @param object $newStatus The status from the engine.
     */
    updateStatus(newStatus) {
        if (newStatus === "error") {
            this.icon.class("error").data("tooltip", "Engine status is <strong>indeterminable</strong>");
        } else {
            this.status = newStatus;
            if (this.node !== undefined) {
                this.updateNodeStatus();
            }
        }
    }

    /**
     * This is called to update the relevant portions of an existing node.
     */
    updateNodeStatus() {
        let icon = this.node.find(E.getCustomTag("statusIcon")),
            version = this.node.find(E.getCustomTag("statusVersion")),
            uptime = this.node.find(E.getCustomTag("statusUptime")),
            gpuName = this.node.find(E.getCustomTag("statusGpuName")),
            gpuTemp = this.node.find(E.getCustomTag("statusGpuTemp")),
            gpuLoad = this.node.find(E.getCustomTag("statusGpuLoad")),
            gpuMemory = this.node.find(E.getCustomTag("statusGpuMemory"));
        
        this.setStatusNodeDetails(icon, version, uptime, gpuName, gpuTemp, gpuLoad, gpuMemory);
    }

    /**
     * Sets the details of each node with the relevant contents from this view
     * @param DOMElement $icon
     * @param DOMElement $version
     * @param DOMElement $uptime
     * @param DOMElement $gpuName
     * @param DOMElement $gpuTemp
     * @param DOMElement $gpuLoad
     * @param DOMElement $gpuMemory
     */
    setStatusNodeDetails(icon, version, uptime, gpuName, gpuTemp, gpuLoad, gpuMemory) {
        let temperatureColor = statusColorScale.get((this.gpuTemp - 50.0) / 50.0).join(","),
            loadColor = statusColorScale.get(this.gpuLoad / 100.0).join(","),
            memoryColor = statusColorScale.get(this.gpuMemoryUtil / 100.0),
            memoryColorString = `rgba(${memoryColor.join(",")}, 0.6)`;
        
        icon.class(this.label).data("tooltip", `Engine status is <strong>${this.label}</strong>`);
        version.show().content(this.version);
        uptime.show().content(this.uptime);
        gpuName.show().content(this.gpuName);
        gpuTemp.show().content(`${this.gpuTemp}`).css("color", `rgb(${temperatureColor})`);
        gpuLoad.show().content(`${this.gpuLoad}`).css("color", `rgb(${loadColor})`);
        gpuMemory.show().content(this.gpuMemoryUsage).css({
            "background-image": `linear-gradient(to right, ${memoryColorString} 0%, ${memoryColorString} ${this.gpuMemoryUtil}%, transparent calc(${this.gpuMemoryUtil}% + 1px))`
        });
    }

    /**
     * Builds the node.
     */
    async build() {
        let node = await super.build();

        let icon = E.statusIcon().on("click", () => this.statusClicked()),
            version = E.statusVersion(),
            uptime = E.statusUptime(),
            gpuName = E.statusGpuName(),
            gpuTemp = E.statusGpuTemp(),
            gpuLoad = E.statusGpuLoad(),
            gpuMemory = E.statusGpuMemory();
            
        if (isEmpty(this.status)) {
            icon.addClass("unknown").data("tooltip", "Engine status is <strong>unknown</strong>");
            version.hide();
            uptime.hide();
            gpuName.hide();
            gpuTemp.hide();
            gpuLoad.hide();
            gpuMemory.hide();
        } else {
            this.setStatusNodeDetails(icon, version, uptime, gpuName, gpuTemp, gpuLoad, gpuMemory);
        }
        
        return node.content(icon ,version, uptime, gpuName, gpuTemp, gpuLoad, gpuMemory);
    }
}

export { StatusView };
