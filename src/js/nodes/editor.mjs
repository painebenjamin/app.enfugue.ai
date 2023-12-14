/** @module nodes/editor */
import { isEmpty, deepClone, sleep } from '../base/helpers.mjs';
import { View } from '../view/base.mjs';
import { FormView } from '../forms/base.mjs';
import { InputView } from '../forms/input.mjs';
import { ElementBuilder } from '../base/builder.mjs';
import { SimpleNotification } from "../common/notify.mjs";
import {
    NodeEditorDecorationsView,
    NodeConnectionSpline
} from './decorations.mjs';
import { NodeView } from './base.mjs';

const E = new ElementBuilder({
    node: "enfugue-node",
    nodeCanvas: 'enfugue-node-canvas',
    editorPosition: 'enfugue-node-editor-position',
    positionReadout: 'enfugue-node-editor-position-readout',
    positionReset: 'enfugue-node-editor-position-reset',
    editorZoom: 'enfugue-node-editor-zoom',
    zoomIn: 'enfugue-node-editor-zoom-in',
    zoomOut: 'enfugue-node-editor-zoom-out',
    zoomReadout: 'enfugue-node-editor-zoom-readout',
    zoomReset: 'enfugue-node-editor-zoom-reset'
});

/**
 * The NodeEditorView is responsible for all interfaces that allows for moving, 
 * zooming, panning, etc. of nodes on some form of canvas.
 */
class NodeEditorView extends View {
    /**
     * @var string The element tag name
     */
    static tagName = 'enfugue-node-editor';

    /**
     * @var int|string The width, either 'auto' or a specific width
     */
    static canvasWidth = 'auto';
    
    /**
     * @var int|string The height, either 'auto' or a specific height
     */
    static canvasHeight = 'auto';

    /**
     * @var float When zooming enabled, this is the maximum zoom in
     */
    static maximumZoom = 2.0;
    
    /**
     * @var float When zooming enabled, this is the maximum zoom out
     */
    static minimumZoom = 0.125;

    /**
     * @var float When zooming enabled, this is the zoom to add per scroll. 
     *            This will be multiplied at high zoom levels.
     */
    static zoomPerScroll = 0.125;

    /**
     * @var bool Whether or not the entire canvas can panned in-place.
     */
    static canMove = true;

    /**
     * @var bool Whether or not zoom is enabled.
     */
    static canZoom = true;

    /**
     * @var bool When true, use (width/2, height/2) as root. When false, use (0, 0).
     */
    static centered = false;

    /**
     * @var string The default cursor mode 
     */
    static defaultCursor = 'default';

    /**
     * @var bool Whether or not to disable all cursor events.
     */
    static disableCursor = false;

    /**
     * @var array<class> All supported node classes. Used when re-instantiating from static data.
     */
    static nodeClasses = [NodeView];

    /**
     * @var array<string> Any number of classes
     */
    static classList = ['loader-cover'];

    /**
     * @var string The icon for zooming in
     */
    static zoomInIcon = 'fa-solid fa-magnifying-glass-plus';

    /**
     * @var string The icon for zooming out
     */
    static zoomOutIcon = 'fa-solid fa-magnifying-glass-minus';

    /**
     * @var bool Default enable bringing to front on focus
     */
    static bringToFrontOnFocus = true;

    /**
     * @param object $config The configuration object.
     * @param ?int windowWidth optional, the expected window width.
     * @param ?int windowHeight optional, the expected window height.
     */
    constructor(config, windowWidth, windowHeight) {
        super(config);

        this.zoom = 1.0;
        if (this.constructor.centered) {
            if (this.constructor.canvasWidth != 'auto') {
                this.left = -this.constructor.canvasWidth / 2;
                if (!isEmpty(windowWidth)) {
                    this.left += windowWidth / 2;
                }
            }
            if (this.constructor.canvasHeight != 'auto') {
                this.top = -this.constructor.canvasHeight / 2;
                if (!isEmpty(windowHeight)) {
                    this.top += windowHeight / 2;
                }
            }
        } else {
            this.left = 0;
            this.top = 0;
        }

        this.nodes = [];
        this.nodeClasses = [].concat(this.constructor.nodeClasses);
        this.nodeFocusCallbacks = [];
        this.nodeCopyCallbacks = [];
        this.setDimensionCallbacks = [];

        this.decorations = new NodeEditorDecorationsView(
            config,
            this,
            this.constructor.canvasWidth,
            this.constructor.canvasHeight
        );

        this.resizeCallbacks = [];
        window.addEventListener('resize', (e) => this.windowResized(e));
    }

    /**
     * Gets a unique name for a node, adding numbers if needed.
     *
     * @param string $name The name of the node.
     */
    getUniqueNodeName(name) {
        let currentName = name,
            currentNames = this.nodes.map((node) => node.getName()),
            duplicates = 1;

        while (currentNames.indexOf(currentName) !== -1) {
            currentName = `${name} ${++duplicates}`;
        }
        return currentName;
    }

    /**
     * @param callable $callback A callback to perform when the window is resized
     */
    onWindowResize(callback) {
        this.resizeCallbacks.push(callback);
    }

    /**
     * @param callable $callback A callback to perform when a node is focused
     */
    onNodeFocus(callback) {
        this.nodeFocusCallbacks.push(callback);
    }

    /**
     * @param callable $callback A callback to perform when a node is copied
     */
    onNodeCopy(callback) {
        this.nodeCopyCallbacks.push(callback);
    }

    /**
     * Called when the window is resized.
     */
    async windowResized() {
        if (this.node !== undefined && !isEmpty(this.node.element)) {
            let [containerWidth, containerHeight] = [
                this.node.element.parentElement.clientWidth,
                this.node.element.parentElement.clientHeight
            ];
            if (containerWidth < this.width) {
                this.node.addClass('oversize-x');
            } else {
                this.node.removeClass('oversize-x');
            }

            if (containerHeight < this.height) {
                this.node.addClass('oversize-y');
            } else {
                this.node.removeClass('oversize-y');
            }
        }

        for (let callback of this.resizeCallbacks) {
            callback();
        }
    }

    /**
     * Adds a node class to the class array after init.
     * @var class $nodeClass The node class to add
     */
    addNodeClass(nodeClass) {
        if (
            this.nodeClasses
                .map((existingNodeClass) => existingNodeClass.name)
                .indexOf(nodeClass.name) === -1
        ) {
            this.nodeClasses.push(nodeClass);
        }
    }

    /**
     * @return int The configured or actual height, depending on current DOM state
     */
    get height() {
        if (this.constructor.canvasHeight == 'auto') {
            if (this.node !== undefined) {
                return this.node.element.clientHeight;
            }
            return 0;
        }
        return this.canvasHeight === undefined
            ? this.constructor.canvasHeight
            : this.canvasHeight;
    }

    /**
     * @return int The configured or actual weight, depending on current DOM state
     */
    get width() {
        if (this.constructor.canvasWidth == 'auto') {
            if (this.node !== undefined) {
                return this.node.element.clientWidth;
            }
            return 0;
        }
        return this.canvasWidth === undefined
            ? this.constructor.canvasWidth
            : this.canvasWidth;
    }

    /**
     * Sets a new height for this editor.
     * @param int $newHeight The new height to set.
     */
    set height(newHeight) {
        this.canvasHeight = newHeight;
        if (this.node !== undefined) {
            let nodeCanvas = this.node.find(E.getCustomTag('nodeCanvas'));
            nodeCanvas.height(newHeight).css('height', `${newHeight}px`);
            this.decorations.setDimension(this.width, newHeight);
            for (let node of this.nodes) {
                node.resetDimension();
            }
        }
    }

    /**
     * Sets a new width for this editor.
     * @param int $newWidth The new width to set.
     */
    set width(newWidth) {
        this.canvasWidth = newWidth;
        if (this.node !== undefined) {
            let nodeCanvas = this.node.find(E.getCustomTag('nodeCanvas'));
            nodeCanvas.width(newWidth).css('width', `${newWidth}px`);
            this.decorations.setDimension(newWidth, this.height);
            for (let node of this.nodes) {
                node.resetDimension();
            }
        }
    }

    /**
     * Adds a callback when dimensions are set
     * @param callable $callback The function to execute
     */
    onSetDimension(callback) {
        this.setDimensionCallbacks.push(callback);
    }

    /**
     * Sets a new width and height for this editor.
     * @param int $newWidth The new width to set.
     * @param int $newHeight The new height to set.
     * @param bool $resetNodes Whether or not to reset the dimensions of the nodes on this canvas.
     */
    setDimension(newWidth, newHeight, resetNodes = true, triggerCallbacks = false) {
        if (isEmpty(newWidth)){
            newWidth = this.canvasWidth;
        }
        if (isEmpty(newHeight)) {
            newHeight = this.canvasHeight;
        }
        this.canvasWidth = newWidth;
        this.canvasHeight = newHeight;
        if (this.node !== undefined) {
            let nodeCanvas = this.node.find(E.getCustomTag('nodeCanvas'));
            nodeCanvas.width(newWidth).height(newHeight).css({
                "width": `${newWidth}px`,
                "height": `${newHeight}px`
            });
            this.decorations.setDimension(newWidth, newHeight);
            if (resetNodes) {
                for (let node of this.nodes) {
                    node.resetDimension();
                }
            }
        }
        this.checkResetCanvasPosition();
        if (triggerCallbacks) {
            for (let callback of this.setDimensionCallbacks) {
                callback(newWidth, newHeight);
            }
        }
    }

    /**
     * Gets the data from the nodes on the canvas
     */
    getState() {
        return this.nodes.map(
            (node) => node.getState.apply(node, Array.from(arguments))
        );
    }

    /**
     * Calls callbacks for when a node is moved
     * @param Node $movedNode The node that was moved.
     */
    nodeMoved(movedNode) {
        // TODO
    }

    /**
     * Calls callbacks for when a node is placed (released somewhere or programmatically set)
     * @param Node $movedNode The node that was placed.
     */
    nodePlaced(node) {
        this.nodeMoved(node);
        // TODO
    }

    /**
     * Gets the node class from the name
     */
    getNodeClass(className) {
        let nodeClass = this.nodeClasses
            .filter((c) => c.name === className)
            .shift();
        if (nodeClass === undefined) {
            throw `Class name out of scope: ${className}`;
        }
        return nodeClass;
    }

    /**
     * Sets new data for the nodes on the canvas
     * @param array<object> $nodes The node data as pulled from getState()
     */
    async setState(nodes) {
        for (let node of this.nodes) {
            this.removeNode(node);
        }
        let canvas = this.node.find(E.getCustomTag("nodeCanvas"));
        for (let node of nodes) {
            let nodeClass = this.getNodeClass(node.classname),
                newNode = new nodeClass(this);
            this.nodes.push(newNode);
            await newNode.setState(node);
            canvas.append(await newNode.getNode());
        }
        this.nodes = this.nodes.map((v, i) => {
            v.index = i;
            return v;
        });
        await this.redraw();
    }

    /**
     * Call to redraw the decorations on the page.
     */
    redraw() {
        if (isEmpty(this.redrawPromise)) {
            this.redrawPromise = new Promise(async (resolve, reject) => {
                for (let node of this.nodes) {
                    await node.resetDimension();
                }
                window.requestAnimationFrame(() => {
                    this.decorations.draw();
                    this.redrawPromise = null;
                    resolve();
                });
            });
        }
        return this.redrawPromise;
    }

    /**
     * Add a node.
     * @param class $nodeClass The class to instantiate.
     * @param any ...arguments to pass to the constructor     
     */
    async addNode(nodeClass) {
        let newNode;
        if (typeof nodeClass === "function") {
            newNode = new nodeClass(this, ...Array.from(arguments).slice(1));
        } else {
            newNode = nodeClass;
            nodeClass.editor = this;
        }
        this.nodes.push(newNode);
        this.nodes = this.nodes.map((v, i) => {
            v.index = i;
            return v;
        });

        if (this.node !== undefined) {
            let childNode = await newNode.getNode(),
                canvas = this.node.find(E.getCustomTag('nodeCanvas'));

            canvas.append(childNode);
        }

        return newNode;
    }

    /**
     * Removes a node from the editor.
     * @param object node The node to remove.
     */
    removeNode(node) {
        for (let childNode of this.nodes) {
            if (childNode == node) {
                this.nodes = this.nodes
                    .filter((n) => n != node)
                    .map((v, i) => {
                        v.index = i;
                        return v;
                    });
                if (!isEmpty(this.node)) {
                    this.node
                        .find(E.getCustomTag('nodeCanvas'))
                        .remove(childNode.node);
                }
                return;
            }
        }
        throw 'Could not find node to remove.';
    }

    /**
     * Triggers callbacks for node focus.
     */
    async focusNode(node) {
        for (let childNode of this.nodes) {
            if (childNode === node && this.constructor.bringToFrontOnFocus) {
                childNode.addClass("focused");
            } else {
                childNode.removeClass("focused");
            }
        }
        for (let focusCallback of this.nodeFocusCallbacks) {
            await focusCallback(node);
        }
    }

    /**
     * Re-orders a node.
     */
    reorderNode(index, node) {
        let currentNodeIndex = this.nodes.indexOf(node);
        if (currentNodeIndex === -1) {
            console.error("Couldn't reorder node, not found in array.");
            return;
        }
        this.nodes = this.nodes.slice(0, currentNodeIndex).concat(this.nodes.slice(currentNodeIndex + 1));
        this.nodes.splice(index, 0, node);
        let nodeCanvas = this.node.find(E.getCustomTag("nodeCanvas"));
        nodeCanvas.remove(node.node);
        if (index > currentNodeIndex) {
            nodeCanvas.insert(index + 4, node.node);
        } else {
            nodeCanvas.insert(index + 3, node.node);
        }
    }

    /**
     * Copies an entire node.
     * @param object $node The node to copy.
     */
    async copyNode(node) {
        let data = node.getState(),
            newNode = await this.addNode(node.constructor);
        data.name += " (copy)";
        data.x += node.constructor.padding;
        data.y += node.constructor.padding;
        await newNode.setState(data);
        for (let copyCallback of this.nodeCopyCallbacks) {
            await copyCallback(newNode, node);
        }
        this.focusNode(newNode);
        return newNode;
    }

    /**
     * Resets position and zoom.
     */
    resetCanvasPosition() {
        if (!isEmpty(this.node)) {
            this.node.find(E.getCustomTag("zoomReset")).trigger("click");
        }
    }

    /**
     * Checks if the canvas is on-screen, then resets the position if not.
     */
    checkResetCanvasPosition() {
        if (!isEmpty(this.node)) {
            let editorPosition = this.node.element.getBoundingClientRect(),
                canvasPosition = this.node.find(E.getCustomTag("nodeCanvas")).element.getBoundingClientRect(),
                intersectLeft = Math.max(editorPosition.x, canvasPosition.x),
                intersectTop = Math.max(editorPosition.y, canvasPosition.y),
                intersectRight = Math.min(
                    (editorPosition.x + editorPosition.width),
                    (canvasPosition.x + canvasPosition.width)
                ),
                intersectBottom = Math.min(
                    (editorPosition.y + editorPosition.height - 100),
                    (canvasPosition.y + canvasPosition.height)
                ),
                editorArea = editorPosition.width * editorPosition.height,
                canvasArea = canvasPosition.width * canvasPosition.height,
                intersectArea = 0.0;

            if (intersectLeft <= intersectRight && intersectTop <= intersectBottom) {
                intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop);
            }

            let intersectAmount = intersectArea / Math.min(canvasArea, editorArea);
            if (intersectAmount <= 0.1) {
                SimpleNotification.notify("Resetting canvas position");
                this.resetCanvasPosition();
            }
        }
    }

    /**
     * The build function creates nodes and binds handlers.
     */
    async build() {
        let node = await super.build(),
            canvas = E.nodeCanvas().css(
                'cursor',
                this.constructor.defaultCursor
            ),
            positionReadout = E.positionReadout().content('0,0'),
            positionReset = E.positionReset().content('Reset'),
            position = E.editorPosition().content(
                positionReadout,
                positionReset
            ),
            zoomReadout = E.zoomReadout().content(`${this.zoom.toFixed(3)}`),
            zoomReset = E.zoomReset().content('Reset'),
            zoomIn = E.zoomIn().content(
                E.i().class(this.constructor.zoomInIcon)
            ),
            zoomOut = E.zoomOut().content(
                E.i().class(this.constructor.zoomOutIcon)
            ),
            zoom = E.editorZoom().content(
                zoomIn,
                zoomOut,
                zoomReset,
                zoomReadout
            );

        if (this.constructor.canvasWidth == 'auto') {
            canvas.css('width', '100%');
        } else {
            canvas.width(this.width).css('width', `${this.width}px`);
        }

        if (this.constructor.canvasHeight == 'auto') {
            canvas.css('height', '100%');
        } else {
            canvas.height(this.height).css('height', `${this.height}px`);
        }

        canvas.css('left', `${this.left}px`);
        canvas.css('top', `${this.top}px`);

        this.decorations.setDimension(this.width, this.height);
        canvas.append(await this.decorations.getNode());

        for (let childNode of this.nodes) {
            canvas.append(await childNode.getNode());
        }

        if (window.innerWidth < this.width) {
            node.addClass('oversize-x');
        }
        if (window.innerHeight < this.height) {
            node.addClass('oversize-y');
        }

        node.append(canvas);

        if (this.constructor.disableCursor) {
            node.css('pointer-events', 'none');
        } else {
            if (this.constructor.canMove) {
                node.append(position);
                let isMoving = false;
                canvas.on('mousedown,touchstart', (e) => {
                    if (e.type === "mousedown" && (!(e.which === 2 || (e.which === 1 && (e.ctrlKey || e.altKey || e.metaKey))) || isMoving)) {
                        return;
                    }
                    e.preventDefault();
                    e.stopPropagation();

                    let startDownX,
                        startDownY,
                        newX = this.left,
                        newY = this.top;

                    if (e.touches) {
                        startDownX = e.touches[0].clientX;
                        startDownY = e.touches[0].clientY;
                    } else {
                        startDownX = e.clientX;
                        startDownY = e.clientY;
                    }

                    isMoving = true;

                    canvas.css('cursor', 'grabbing');
                    canvas
                        .on('mousemove,touchmove', (e2) => {
                            e2.preventDefault();
                            let deltaX, deltaY;
                            if (e2.touches) {
                                deltaX = e2.touches[0].clientX - startDownX;
                                deltaY = e2.touches[0].clientY - startDownY;
                            } else {
                                deltaX = e2.clientX - startDownX;
                                deltaY = e2.clientY - startDownY;
                            }
                            newX = this.left + deltaX;
                            newY = this.top + deltaY;

                            canvas.css({
                                left: `${newX}px`,
                                top: `${newY}px`
                            });

                            let canvasReadoutX, canvasReadoutY;

                            if (this.constructor.centered) {
                                let canvasCenterX = (this.width / 2) - (node.element.clientWidth / 2),
                                    canvasCenterY = (this.height / 2) - (node.element.clientHeight / 2);
                                
                                canvasReadoutX = -canvasCenterX - newX / this.zoom,
                                canvasReadoutY = -canvasCenterY - newY / this.zoom;
                            } else {
                                canvasReadoutX = -newX / this.zoom;
                                canvasReadoutY = -newY / this.zoom;
                            }

                            if (this.zoom != 1.0) {
                                canvasReadoutX = canvasReadoutX.toFixed(2);
                                canvasReadoutY = canvasReadoutY.toFixed(2);
                            }
                            positionReadout.content(`${canvasReadoutX},${canvasReadoutY}`);
                        })
                        .on('mouseup,mouseleave,touchend,touchleave', (e2) => {
                            e2.preventDefault();
                            e2.stopPropagation();

                            if (e2.touches) {
                                this.left = newX;
                                this.top = newY;
                            } else {
                                let deltaX, deltaY;
                                deltaX = e2.clientX - startDownX;
                                deltaY = e2.clientY - startDownY;

                                this.left += deltaX;
                                this.top += deltaY;
                            }

                            canvas.off('mouseup,mousemove,mouseleave,touchend,touchleave');
                            canvas.css({
                                left: `${this.left}px`,
                                top: `${this.top}px`,
                                cursor: this.constructor.defaultCursor
                            });
                            
                            let canvasReadoutX, canvasReadoutY;

                            if (this.constructor.centered) {
                                let canvasCenterX = (this.width / 2) - (node.element.clientWidth / 2),
                                    canvasCenterY = (this.height / 2) - (node.element.clientHeight / 2);
                                
                                canvasReadoutX = -canvasCenterX - this.left / this.zoom,
                                canvasReadoutY = -canvasCenterY - this.top / this.zoom;
                            } else {
                                canvasReadoutX = -this.left / this.zoom;
                                canvasReadoutY = -this.top / this.zoom;
                            }

                            if (this.zoom != 1.0) {
                                canvasReadoutX = canvasReadoutX.toFixed(2);
                                canvasReadoutY = canvasReadoutY.toFixed(2);
                            }
                            positionReadout.content(`${canvasReadoutX},${canvasReadoutY}`);
                            isMoving = false;
                        });
                });

                positionReset.on('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    if (this.constructor.centered) {
                        this.left = -(this.width / 2) * this.zoom + node.element.clientWidth / 2;
                        this.top =  -(this.height / 2) * this.zoom + node.element.clientHeight / 2;
                    } else {
                        this.left = 0;
                        this.top = 0;
                    }

                    positionReadout.content('0,0');

                    canvas.css({
                        left: `${this.left}px`,
                        top: `${this.top}px`
                    });
                });
            }

            if (this.constructor.canZoom) {
                node.append(zoom);
                let setZoom = (nextZoom, zoomX, zoomY) => {
                    // When you zoom, we want your window to keep the point you zoomed in
                    // on under your mouse cursor - so essentiall we scale about that point.
                    let scaleChange = nextZoom - this.zoom,
                        offsetX = -(zoomX * scaleChange),
                        offsetY = -(zoomY * scaleChange);

                    this.left += offsetX;
                    this.top += offsetY;
                    this.zoom = nextZoom;

                    zoomReadout.content(this.zoom.toFixed(3));
                    canvas.css({
                        left: `${this.left}px`,
                        top: `${this.top}px`,
                        transform: `scale(${this.zoom})`
                    });

                    if (this.constructor.canMove) {
                        let canvasReadoutX, canvasReadoutY;

                        if (this.constructor.centered) {
                            let canvasCenterX = (this.width / 2) - (node.element.clientWidth / 2),
                                canvasCenterY = (this.height / 2) - (node.element.clientHeight / 2);
                            
                            canvasReadoutX = -canvasCenterX - this.left / this.zoom,
                            canvasReadoutY = -canvasCenterY - this.top / this.zoom;
                        } else {
                            canvasReadoutX = -this.left / this.zoom;
                            canvasReadoutY = -this.top / this.zoom;
                        }

                        if (this.zoom != 1.0) {
                            canvasReadoutX = canvasReadoutX.toFixed(2);
                            canvasReadoutY = canvasReadoutY.toFixed(2);
                        }
                        positionReadout.content(`${canvasReadoutX},${canvasReadoutY}`);
                    }

                    if (this.zoom > 1) {
                        node.addClass("zoom-in");
                    } else {
                        node.removeClass("zoom-in");
                    }
                    if (this.zoom < 1) {
                        node.addClass("zoom-out");
                    } else {
                        node.removeClass("zoom-out");
                    }
                };
                canvas.on('wheel', (e) => {
                    if (e.target.tagName !== 'ENFUGUE-NODE-CANVAS') {
                        // Figure out if anything scrolled
                        let scrollDown = e.deltaY > 0, target = e.target;
                        while (true) {
                            if (target === undefined || target === null || target.tagName === 'ENFUGUE-NODE-EDITOR' || target.tagName === 'ENFUGUE-NODE-CANVAS') {
                                break;
                            }
                            if (target.scrollHeight !== target.offsetHeight && target.tagName !== 'ENFUGUE-NODE-CONTENTS') {
                                let currentBottom = target.offsetHeight + target.scrollTop,
                                    targetOverflow = getComputedStyle(target).overflowY;
                                if (["auto", "scroll"].indexOf(targetOverflow) !== -1) {
                                    if (target.scrollHeight - currentBottom > 1 && scrollDown) {
                                        // Element scrolled
                                        return;
                                    } else if (target.scrollTop > 0 && !scrollDown) {
                                        // Element scrolled
                                        return;
                                    }
                                }
                            }
                            target = target.parentElement;
                        }
                    }

                    e.preventDefault();
                    e.stopPropagation();
                    let nextZoom, zoomMultiplier = 1;
                    
                    if (this.zoom >= 5.0) {
                        zoomMultiplier = 4;
                    } else if (this.zoom >= 2.5) {
                        zoomMultiplier = 2;
                    }

                    if (e.deltaY < 0) {
                        nextZoom = this.zoom + (this.constructor.zoomPerScroll * zoomMultiplier);
                    } else {
                        nextZoom = this.zoom - (this.constructor.zoomPerScroll * zoomMultiplier);
                    }

                    let target = e.target,
                        zoomX = e.offsetX,
                        zoomY = e.offsetY;

                    while (
                        target.tagName != 'ENFUGUE-NODE-CANVAS' &&
                        target.tagName != 'ENFUGUE-NODE-EDITOR-CANVAS'
                    ) {
                        zoomX += target.offsetLeft;
                        zoomY += target.offsetTop;
                        target = target.parentElement;
                    }

                    if (
                        nextZoom >= this.constructor.minimumZoom &&
                        nextZoom <= this.constructor.maximumZoom &&
                        nextZoom != this.zoom
                    ) {
                        setZoom(nextZoom, zoomX, zoomY);
                    }
                });

                zoomIn
                    .on('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        let nextZoom = this.zoom + this.constructor.zoomPerScroll;
                        if (
                            nextZoom >= this.constructor.minimumZoom &&
                            nextZoom <= this.constructor.maximumZoom &&
                            nextZoom != this.zoom
                        ) {
                            setZoom(nextZoom, this.width / 2, this.height / 2);
                        }
                    })
                    .on('dblclick', (e) => {
                        e.preventDefault();
                    });

                zoomOut
                    .on('click', (e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        let nextZoom = this.zoom - this.constructor.zoomPerScroll;
                        if (
                            nextZoom >= this.constructor.minimumZoom &&
                            nextZoom <= this.constructor.maximumZoom &&
                            nextZoom != this.zoom
                        ) {
                            setZoom(nextZoom, this.width / 2, this.height / 2);
                        }
                    })
                    .on('dblclick', (e) => {
                        e.preventDefault();
                    });

                zoomReset.on('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();

                    this.zoom = 1.0;
                    zoomReadout.content('1.000');

                    canvas.css({
                        transform: 'scale(1)'
                    });

                    positionReset.trigger('click');
                });
            }
        }
        return node;
    }
}

/**
 * The 'NodeView' class is a simple extension of the Editor 
 * that disables editing.
 */
class NodesView extends NodeEditorView {
    static tagName = 'enfugue-nodes';
    static canMove = false;
    static canZoom = false;
}

export { NodeEditorView, NodesView };
