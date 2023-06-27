/** @module view/scribble */
import { View, ParentView } from './base.mjs';
import { ElementBuilder } from '../base/builder.mjs';
import { isEmpty } from '../base/helpers.mjs';

const E = new ElementBuilder();

class ScribbleView extends View {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-scribble-view";

    /**
     * @var int The default pencil size
     */
    static defaultPencilSize = 6;

    /**
     * @var int The maximum pencil size
     */
    static maximumPencilSize = 100;

    /**
     * @var string The default pencil shape
     */
    static defaultPencilShape = "circle";

    /**
     * Allows for a simple 'scribble' interface, a canvas that can be painted on in pure white/black.
     */
    constructor(config, width, height) {
        super(config);
        this.width = width;
        this.height = height;
        this.active = false;

        this.shape = this.constructor.defaultPencilShape;
        this.size = this.constructor.defaultPencilSize;
        this.isEraser = false;

        this.memoryCanvas = document.createElement("canvas");
        this.visibleCanvas = document.createElement("canvas");

        if (!isEmpty(width) && !isEmpty(height)) {
            this.memoryCanvas.width = width;
            this.memoryCanvas.height = height;
            this.visibleCanvas.width = width;
            this.visibleCanvas.height = height;
        }
    }

    /**
     * Gets the canvas image as a data URL.
     * We use the visible canvas so that we crop appropriately.
     */
    get src() {
        this.updateVisibleCanvas();
        return this.visibleCanvas.toDataURL();
    }

    /**
     * Clears the canvas in memory.
     */
    clearMemory() {
        let memoryContext = this.memoryCanvas.getContext("2d");
        memoryContext.clearRect(0, 0, this.memoryCanvas.width, this.memoryCanvas.height);
        this.updateVisibleCanvas();
    }

    /**
     * Sets the canvas memory from an image
     */
    setMemory(newImage) {
        let newMemoryCanvas = document.createElement("canvas");
        newMemoryCanvas.width = newImage.width;
        newMemoryCanvas.height = newImage.height;

        this.visibleCanvas.width = newMemoryCanvas.width;
        this.visibleCanvas.height = newMemoryCanvas.height;

        let memoryContext = newMemoryCanvas.getContext("2d");
        memoryContext.drawImage(newImage, 0, 0);
        
        this.memoryCanvas = newMemoryCanvas;
        this.updateVisibleCanvas();
    }

    /**
     * Trigger resize on the canvas.
     */
    resizeCanvas(width, height) {
        this.width = width;
        this.height = height;
        this.visibleCanvas.width = width;
        this.visibleCanvas.height = height;
        
        if (width > this.memoryCanvas.width || height > this.memoryCanvas.height) {
            let newMemoryCanvas = document.createElement("canvas");
            newMemoryCanvas.width = width;
            newMemoryCanvas.height = height;
            let newMemoryContext = newMemoryCanvas.getContext("2d");
            newMemoryContext.drawImage(this.memoryCanvas, 0, 0);
            this.memoryCanvas = newMemoryCanvas;
        }
        this.updateVisibleCanvas();
    }

    /**
     * Updates the visible canvas with the content of the memory canvas.
     */
    updateVisibleCanvas() {
        let canvasContext = this.visibleCanvas.getContext("2d");
        canvasContext.beginPath();
        canvasContext.rect(0, 0, this.width, this.height);
        canvasContext.fillStyle = "white";
        canvasContext.fill();
        canvasContext.drawImage(this.memoryCanvas, 0, 0);
    }

    /**
     * Gets the zoom-adjusted x, y coordinates from an event
     */
    getCoordinates(e) {
        return [e.offsetX, e.offsetY];
    }

    /**
     * The 'mouseenter' handler
     */
    onNodeMouseEnter(e) {
        this.active = false;
    }

    /**
     * The 'mouseleave' handler
     */
    onNodeMouseLeave(e) {
        this.active = false;
        this.updateVisibleCanvas();
        this.lastX = null;
        this.lastY = null;
    }
    
    /**
     * The 'mousedown' handler
     */
    onNodeMouseDown(e) {
        if (e.which !== 1) return;
        e.preventDefault();
        e.stopPropagation();
        this.active = true;
        let [eventX, eventY] = this.getCoordinates(e);
        if (!isEmpty(this.lastX) && !isEmpty(this.lastY) && e.shiftKey) {
            this.drawLineTo(eventX, eventY);
        }
        if (e.altKey || this.isEraser) {
            this.erase(eventX, eventY);
        } else {
            this.drawMemory(eventX, eventY);
        }
    }

    /**
     * The 'mouseup' handler
     */
    onNodeMouseUp(e) {
        this.active = false;
    }

    /**
     * The 'mousemove' handler
     */
    onNodeMouseMove(e) {
        let [eventX, eventY] = this.getCoordinates(e);
        if (!isEmpty(this.lastDrawTime) && !e.altKey && !this.isEraser) {
            let timeSinceLastDraw = (new Date()).getTime() - this.lastDrawTime;
            if (timeSinceLastDraw < 50) {
                this.drawLineTo(eventX, eventY);
            }
        }
        if (this.active) {
            e.preventDefault();
            e.stopPropagation();
            if (e.altKey || this.isEraser){
                this.erase(eventX, eventY);
            } else {
                this.drawMemory(eventX, eventY);
            }
        } else {
            this.drawVisible(eventX, eventY);
        }
    }

    /**
     * Decreases the size by 2px
     */
    decreaseSize(){
        this.size = Math.max(2, this.size - 2);
    }

    /**
     * Increases the size by 2px
     */
    increaseSize(){
        this.size = Math.min(this.constructor.maximumPencilSize, this.size + 2);
    }

    /**
     * The 'wheel' handler
     */
    onNodeWheel(e){
        if (e.ctrlKey) {
            e.preventDefault();
            return;
        }
        if (e.deltaY > 0) {
            this.decreaseSize();
        } else {
            this.increaseSize();
        }
        let [eventX, eventY] = this.getCoordinates(e);
        this.drawVisible(eventX, eventY);
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Erases the current tool from the canvas
     */
    erase(x, y) {
        let context = this.memoryCanvas.getContext("2d");
        context.save();
        this.drawPencilShape(context, x, y);
        context.clip();
        context.clearRect(0, 0, this.memoryCanvas.width, this.memoryCanvas.height);
        context.restore();
        this.updateVisibleCanvas();
    }

    /**
     * Draws on the memory canvas from the event, then copies to visible.
     */
    drawMemory(x, y) {
        let context = this.memoryCanvas.getContext("2d");
        this.drawPencilShape(context, x, y);
        context.fillStyle = "#000000";
        context.fill();
        this.updateVisibleCanvas();
        this.lastX = x;
        this.lastY = y;
        this.lastDrawTime = (new Date()).getTime();
    }

    /**
     * Draws only on the visible canvas.
     */
    drawVisible(x, y) {
        this.updateVisibleCanvas();
        let context = this.visibleCanvas.getContext("2d");
        this.size -= 1;
        this.drawPencilShape(context, x, y);
        context.strokeStyle = "#ffffff";
        context.lineWidth = 1;
        context.stroke();
        this.size += 1;
        this.drawPencilShape(context, x, y);
        context.strokeStyle = "#000000";
        context.lineWidth = 1;
        context.stroke();
    }

    /**
     * Draws a line between the last point and the current point.
     */
    drawLineTo(x, y) {
        let context = this.memoryCanvas.getContext("2d");
        context.beginPath();
        context.moveTo(this.lastX, this.lastY);
        context.lineTo(x, y);
        context.strokeStyle = "#000000";
        context.lineWidth = this.size;
        context.stroke();
    }

    /**
     * This is shared between drawMemory and drawVisible; it traces the path the
     * current tool will either stroke or fill.
     */
    drawPencilShape(context, x, y) {
        context.beginPath();
        if (this.shape === "circle") {
            context.arc(x, y, this.size / 2, 0, 2 * Math.PI);
        } else {
            let left = Math.max(0, x - this.size / 2),
                top = Math.max(0, y - this.size / 2),
                right = Math.min(left + this.size, this.width),
                bottom = Math.min(top + this.size, this.height);
            context.moveTo(left, top);
            context.lineTo(right, top);
            context.lineTo(right, bottom);
            context.lineTo(left, bottom);
            context.lineTo(left, top);
        }
    };

    /**
     * On build, append canvas.
     */
    async build() {
        let node = await super.build();
        node.append(this.visibleCanvas);
        node.on("dblclick", (e) => { e.preventDefault(); e.stopPropagation(); });
        node.on("mouseenter", (e) => this.onNodeMouseEnter(e));
        node.on("mousemove", (e) => this.onNodeMouseMove(e));
        node.on("mousedown", (e) => this.onNodeMouseDown(e));
        node.on("mouseup", (e) => this.onNodeMouseUp(e));
        node.on("mouseleave", (e) => this.onNodeMouseLeave(e));
        node.on("wheel", (e) => this.onNodeWheel(e));
        this.updateVisibleCanvas();
        return node;
    }
};

export { ScribbleView };
