/** @module view/vector */
import { View } from "./base.mjs";
import { Point, Drawable, Vector } from "../graphics/geometry.mjs";
import { Spline, SplinePoint } from "../graphics/spline.mjs";
import { ElementBuilder } from '../base/builder.mjs';
import {
    isEmpty,
    deepClone,
    isEquivalent,
    bindPointerUntilRelease,
    getPointerEventCoordinates,
} from '../base/helpers.mjs';

const E = new ElementBuilder();

class EditableSplineVector extends Spline {
    /**
     * @var int Half handle width
     */
    static handleWidth = 10;

    /**
     * @var int Half arrow length
     */
    static arrowLength = 10;

    /**
     * Gets the handle drawable about a point
     */
    getHandleAboutPoint(point, offsetAmount = 0) {
        return new Drawable([
            new Point(
                point.x-this.constructor.handleWidth-offsetAmount,
                point.y-this.constructor.handleWidth-offsetAmount,
            ),
            new Point(
                point.x+this.constructor.handleWidth+offsetAmount,
                point.y-this.constructor.handleWidth-offsetAmount,
            ),
            new Point(
                point.x+this.constructor.handleWidth+offsetAmount,
                point.y+this.constructor.handleWidth+offsetAmount,
            ),
            new Point(
                point.x-this.constructor.handleWidth-offsetAmount,
                point.y+this.constructor.handleWidth+offsetAmount,
            ),
            new Point(
                point.x-this.constructor.handleWidth-offsetAmount,
                point.y-this.constructor.handleWidth-offsetAmount,
            ),
        ]);
    }

    /**
     * Gets a quadratic point along a curve with time t
     */
    quadraticBezierCurvePoint(p0, p1, p2, t) {
        // Ensure t is between 0 and 1
        t = Math.max(0, Math.min(1, t));

        // Calculate the coefficients
        const c0 = (1 - t) * (1 - t);
        const c1 = 2 * (1 - t) * t;
        const c2 = t * t;

        // Calculate the x and y coordinates of the point on the curve
        const x = c0 * p0.x + c1 * p1.x + c2 * p2.x;
        const y = c0 * p0.y + c1 * p1.y + c2 * p2.y;

        return new Point(x, y);
    }

    /**
     * Gets a cubic point along a curve with time t
     */
    cubicBezierCurvePoint(p0, p1, p2, p3, t) {
        // Ensure t is between 0 and 1
        t = Math.max(0, Math.min(1, t));

        // Calculate the coefficients
        const c0 = Math.pow(1 - t, 3);
        const c1 = 3 * Math.pow(1 - t, 2) * t;
        const c2 = 3 * (1 - t) * Math.pow(t, 2);
        const c3 = Math.pow(t, 3);

        // Calculate the x and y coordinates of the point on the curve
        const x = c0 * p0.x + c1 * p1.x + c2 * p2.x + c3 * p3.x;
        const y = c0 * p0.y + c1 * p1.y + c2 * p2.y + c3 * p3.y;

        return new Point(x, y);
    }

    /**
     * Removes the point at the specified index
     */
    removePoint(index) {
        this.points = this.points.slice(0, index).concat(this.points.slice(index+1));
    }

    /**
     * Adds a point at the specified index
     */
    addPoint(index) {
        let start = this.points[index-1],
            end = this.points[index],
            midPoint;

        if (start.pointType === SplinePoint.TYPE_LINEAR && end.pointType === SplinePoint.TYPE_BEZIER) {
            midPoint = this.quadraticBezierCurvePoint(start, end.controlPoint1, end, 0.5);
        } else if (start.pointType === SplinePoint.TYPE_BEZIER && end.pointType === SplinePoint.TYPE_LINEAR) {
            midPoint = this.quadraticBezierCurvePoint(start, start.controlPoint2, end, 0.5);
        } else if (start.pointType !== SplinePoint.TYPE_LINEAR && end.pointType !== SplinePoint.TYPE_LINEAR) {
            midPoint = this.cubicBezierCurvePoint(start, start.controlPoint2, end.controlPoint1, end, 0.5);
        } else {
            midPoint = (new Vector(start, end)).halfway;
        }
        this.points = this.points.slice(0, index)
                                 .concat([new SplinePoint(midPoint, SplinePoint.TYPE_LINEAR)])
                                 .concat(this.points.slice(index));
    }

    /**
     * Gets the angle between points in radians
     */
    getEndingAngleBetweenPoints(start, end) {
        start = start.clone();
        end = end.clone();
        if (start.pointType === SplinePoint.TYPE_LINEAR && end.pointType === SplinePoint.TYPE_BEZIER) {
            start = this.quadraticBezierCurvePoint(start, end.controlPoint1, end, 0.9);
        } else if (start.pointType === SplinePoint.TYPE_BEZIER && end.pointType === SplinePoint.TYPE_LINEAR) {
            start = this.quadraticBezierCurvePoint(start, start.controlPoint2, end, 0.9);
        } else if (start.pointType !== SplinePoint.TYPE_LINEAR && end.pointType !== SplinePoint.TYPE_LINEAR) {
            start = this.cubicBezierCurvePoint(start, start.controlPoint2, end.controlPoint1, end, 0.9);
        }
        let vector = new Vector(start, end);
        return vector.rad + 3*Math.PI/2;
    }

    /**
     * Draws the spline and points
     */
    draw(context, color, drawHandles, selected) {
        context.strokeStyle = color;
        context.fillStyle = color;
        context.lineWidth = 2;
        this.stroke(context);

        let [penultimate, end] = this.points.slice(-2),
            arrowhead = new Drawable([
                new Point(
                    end.x,
                    end.y
                ),
                new Point(
                    end.x - this.constructor.arrowLength,
                    end.y - this.constructor.arrowLength
                ),
                new Point(
                    end.x + this.constructor.arrowLength,
                    end.y - this.constructor.arrowLength
                ),
                new Point(
                    end.x,
                    end.y
                )
            ]);

        arrowhead.rotateAbout(this.getEndingAngleBetweenPoints(penultimate, end), end);
        arrowhead.fill(context);

        if (!drawHandles) return;

        // Color what is selected
        context.fillStyle = "rgba(0,0,0,0.4)";

        for (let pointIndex in this.points) {
            let point = this.points[pointIndex],
                selectedPoints = [];

            if (!isEmpty(selected) && !isEmpty(selected[pointIndex])) {
                selectedPoints = selected[pointIndex];
            }

            let handle = this.getHandleAboutPoint(point);
            context.lineWidth = 2;
            context.strokeStyle = selectedPoints.indexOf("anchorPoint") !== -1
                ? color
                : "#ffffff";
            handle.stroke(context);
            handle.fill(context);
            context.lineWidth = 1;
            if (point.pointType === SplinePoint.TYPE_BEZIER) {
                if (!isEmpty(point.controlPoint1)) {
                    let controlHandle = this.getHandleAboutPoint(point.controlPoint1, -2),
                        controlLine = new Drawable([point.anchorPoint, point.controlPoint1]);
                    context.strokeStyle = selectedPoints.indexOf("controlPoint1") !== -1
                        ? color
                        : "#ffffff";
                    context.setLineDash([]);
                    controlHandle.stroke(context);
                    context.setLineDash([4,2]);
                    controlLine.stroke(context);
                }
                if (!isEmpty(point.controlPoint2)) {
                    let controlHandle = this.getHandleAboutPoint(point.controlPoint2, -2),
                        controlLine = new Drawable([point.anchorPoint, point.controlPoint2]);
                    context.strokeStyle = selectedPoints.indexOf("controlPoint2") !== -1
                        ? color
                        : "#ffffff";
                    context.setLineDash([]);
                    controlHandle.stroke(context);
                    context.setLineDash([4,2]);
                    controlLine.stroke(context);
                }
            }
            context.setLineDash([]);
        }

    }
}

/**
 * Enum for cursor mode
 */
class VectorCursorMode {
    static NONE = 0;
    static MOVE = 1;
    static ROTATE = 2;
    static SELECT = 3;
}

/**
 * Allows drawing input vectors
 */
class VectorView extends View {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-vector-view";

    /**
     * @var int History length
     */
    static historyLength = 250;

    /**
     * On construct, add buffers for splines and callbacks
     */
    constructor(config, width, height) {
        super(config);
        this.width = width;
        this.height = height;
        this.splines = [];
        this.onChangeCallbacks = [];
        this.canvas = document.createElement("canvas");
        this.canvas.width = width;
        this.canvas.height = height;
        this.mode = VectorCursorMode.NONE;
        this.selected = {};
        this.startPosition = null;
        this.startSelected = null;
        this.startRadians = null;
        this.extendCopies = 2;
        this.history = [];
        this.redoStack = [];
    }

    /**
     * Encodes data from a spline
     */
    getValueFromSpline(spline) {
        return spline.points.map((point) => {
            let mapped = {
                "anchor": [point.x, point.y]
            };
            if (point.pointType === SplinePoint.TYPE_BEZIER) {
                if (!isEmpty(point.controlPoint1)) {
                    mapped["control_1"] = [point.cp1x, point.cp1y];
                }
                if (!isEmpty(point.controlPoint2)) {
                    mapped["control_2"] = [point.cp2x, point.cp2y];
                }
            }
            return mapped;
        });
    }

    /**
     * Gets the encoded data
     */
    get value() {
        return this.splines.map((spline) => this.getValueFromSpline(spline));
    }

    /**
     * Gets the encoded data with repetitions
     */
    get extendedValue() {
        return this.splines.map((spline) => {
            let encodedPoints = this.getValueFromSpline(spline);
            for (let j = 0; j < this.extendCopies; j++) {
                let cloned = spline.clone(),
                    splineStart = spline.points[0],
                    [splinePenultimate, splineEnd] = spline.points.slice(-2),
                    deltaX = splineEnd.x - splineStart.x,
                    deltaY = splineEnd.y - splineStart.y;

                cloned.translatePoint(new Point(deltaX, deltaY));
                let clonePoints = this.getValueFromSpline(cloned);
                encodedPoints[encodedPoints.length-1].control_2 = clonePoints[0].control_1;
                encodedPoints = encodedPoints.concat(clonePoints.slice(1));
                spline = cloned;
            }
            return encodedPoints;
        });
    }

    /**
     * Sets the encoded data
     */
    set value(newValue) {
        this.splines = newValue.map((encodedSpline) => {
            return new EditableSplineVector(
                encodedSpline.map((encodedPoint) => {
                    let anchorPoint = new Point(
                            encodedPoint.anchor[0],
                            encodedPoint.anchor[1]
                        ),
                        pointType = isEmpty(encodedPoint.control_1) && isEmpty(encodedPoint.control_2)
                            ? SplinePoint.TYPE_LINEAR
                            : SplinePoint.TYPE_BEZIER,
                        controlPoint1 = isEmpty(encodedPoint.control_1)
                            ? undefined
                            : new Point(
                                encodedPoint.control_1[0],
                                encodedPoint.control_1[1]
                            ),
                        controlPoint2 = isEmpty(encodedPoint.control_2)
                            ? undefined
                            : new Point(
                                encodedPoint.control_2[0],
                                encodedPoint.control_2[1]
                            );
                    return new SplinePoint(
                        anchorPoint,
                        pointType,
                        controlPoint1,
                        controlPoint2
                    );
                })
            );
        });
        this.updateCanvas();
    }

    /**
     * Gets the theme color
     */
    get color() {
        return window.getComputedStyle(document.documentElement).getPropertyValue("--theme-color-primary");
    }

    /**
     * Adds an onchange handler
     */
    onChange(callback) {
        this.onChangeCallbacks.push(callback);
    }

    /**
     * Triggers onChange handlers
     */
    changed(addToHistory=true, resetStack=true) {
        let callbackValue = this.value,
            lastCallbackValue = this.lastCallbackValue;
        if (!isEquivalent(callbackValue, lastCallbackValue)) {
            for (let handler of this.onChangeCallbacks) {
                handler(callbackValue);
            }
            if (addToHistory) {
                this.history = this.history.concat([callbackValue]).slice(-this.constructor.historyLength);
                if (resetStack) {
                    this.redoStack = [];
                }
            }
        }
        this.lastCallbackValue = callbackValue;
    }

    /**
     * Does an undo
     */
    undo() {
        if (!isEmpty(this.history)) {
            let currentState = this.history.pop(-1);
            this.redoStack.push(currentState);
            this.value = this.history[this.history.length-1];
            this.changed(false, false);
        }
    }

    /**
     * Does a redo
     */
     redo() {
        if (!isEmpty(this.redoStack)) {
            let lastState = this.redoStack.pop(-1);
            this.value = lastState;
            this.changed(true, false);
        }
    }

    /**
     * Trigger resize on the canvas.
     */
    resizeCanvas(width, height) {
        this.width = width;
        this.height = height;
        this.canvas.width = width;
        this.canvas.height = height;
        for (let spline of this.splines) {
            for (let point of spline.points) {
                point.x = Math.min(point.x, width);
                point.y = Math.min(point.y, height);
                if (!isEmpty(point.controlPoint1)) {
                    point.cp1x = Math.min(point.cp1x, width);
                    point.cp1y = Math.min(point.cp1y, height);
                }
                if (!isEmpty(point.controlPoint2)) {
                    point.cp2x = Math.min(point.cp2x, width);
                    point.cp2y = Math.min(point.cp2y, height);
                }
            }
        }
        this.updateCanvas();
        this.changed();
    }

    /**
     * Draws the canvas
     */
    updateCanvas() {
        let context = this.canvas.getContext("2d"),
            color = this.color;
        context.clearRect(0, 0, this.width, this.height);
        for (let i in this.splines) {
            let spline = this.splines[i];
            spline.draw(context, color, true, this.selected[i]);
            for (let j = 0; j < this.extendCopies; j++) {
                let cloned = spline.clone(),
                    splineStart = spline.points[0],
                    [splinePenultimate, splineEnd] = spline.points.slice(-2),
                    deltaX = splineEnd.x - splineStart.x,
                    deltaY = splineEnd.y - splineStart.y;

                cloned.translatePoint(new Point(deltaX, deltaY));
                context.setLineDash([2,4]);
                cloned.draw(context, color, false);
                spline = cloned;
            }
            context.setLineDash([]);
        }
    }

    /**
     * Sets the number of copies
     */
    setCopies(newCopies) {
        this.extendCopies = newCopies;
        this.updateCanvas();
    }

    /**
     * Gets the zoom-adjusted x, y coordinates from an event
     */
    getCoordinates(e) {
        if (e.touches && e.touches.length > 0) {
            let frame = e.target.getBoundingClientRect();
            return [
                e.touches[0].clientX - frame.x,
                e.touches[0].clientY - frame.y
            ];
        } else {
            return [
                e.offsetX,
                e.offsetY
            ];
        }
    }

    /**
     * Gets any point under coordinates
     */
    getPointFromCoordinates(x, y) {
        for (let i in this.splines) {
            let spline = this.splines[i];
            for (let j in spline.points) {
                let point = spline.points[j];
                for (let pointName of ["anchorPoint", "controlPoint1", "controlPoint2"]) {
                    if (!isEmpty(point[pointName])) {
                        if (point[pointName].x - EditableSplineVector.handleWidth <= x &&
                            point[pointName].x + EditableSplineVector.handleWidth >  x &&
                            point[pointName].y - EditableSplineVector.handleWidth <= y &&
                            point[pointName].y + EditableSplineVector.handleWidth >  y
                        ) {
                            return [parseInt(i), parseInt(j), pointName];
                        }
                    }
                }
            }
        }
    }

    /**
     * Gets any spline under coordinates
     */
    getSplineFromCoordinates(x, y) {
        let point = new Point(x, y);
        for (let i in this.splines) {
            let spline = this.splines[i],
                splineIndex = spline.pointAlongSpline(point, 10);
            if (!isEmpty(splineIndex)) {
                return [parseInt(i), splineIndex+1];
            }
        }
    }

    /**
     * The 'mousedown' handler
     */
    onNodeMouseDown(e) {
        if (e.type === "mousedown" && e.which !== 1) return;
        if (e.metaKey || (e.ctrlKey && !e.shiftKey)) return;

        e.preventDefault();
        e.stopPropagation();
        let [x, y] = this.getCoordinates(e);
        try {
            let point = this.getPointFromCoordinates(x, y);
            if (!isEmpty(point)) {
                let [splineIndex, pointIndex, pointName] = point;
                if (e.altKey && pointName === "anchorPoint") {
                    if (this.splines[splineIndex].points.length === 2) {
                        // Remove spline
                        this.splines = this.splines.slice(0, splineIndex).concat(this.splines.slice(splineIndex+1));
                    } else {
                        // Remove point in spline
                        this.splines[splineIndex].removePoint(pointIndex);
                    }
                } else {
                    if (!e.shiftKey) {
                        this.selected = {};
                    }
                    if (isEmpty(this.selected[splineIndex])) {
                        this.selected[splineIndex] = {};
                    }
                    if (isEmpty(this.selected[splineIndex][pointIndex])) {
                        this.selected[splineIndex][pointIndex] = [];
                    }
                    let isSelected = this.selected[splineIndex][pointIndex].indexOf(pointName) !== -1;
                    if (!isSelected) {
                        this.selected[splineIndex][pointIndex].push(pointName);
                    }
                    this.mode = VectorCursorMode.MOVE;
                }
                return;
            }

            let spline = this.getSplineFromCoordinates(x, y);
            if (!isEmpty(spline)) {
                let [splineIndex, pointIndex] = spline;
                if (e.altKey) {
                    this.splines[splineIndex].addPoint(pointIndex);
                } else {
                    if (!e.shiftKey) {
                        this.selected = {};
                    }
                    this.selected[splineIndex] = {};
                    for (let pointIndex in this.splines[splineIndex].points) {
                        let selectedPoints = ["anchorPoint"];
                        for (let pointName of ["controlPoint1", "controlPoint2"]) {
                            if (!isEmpty(this.splines[splineIndex].points[pointIndex][pointName])) {
                                selectedPoints.push(pointName);
                            }
                        }
                        this.selected[splineIndex][pointIndex] = selectedPoints;
                    }
                    this.mode = VectorCursorMode.MOVE;
                }
                return;
            }

            if (e.altKey) {
                let newSpline = new EditableSplineVector([
                    new SplinePoint(new Point(x, y), SplinePoint.TYPE_LINEAR),
                    new SplinePoint(new Point(x, y), SplinePoint.TYPE_LINEAR)
                ]);
                this.splines.push(newSpline);
                this.selected = {};
                this.selected[this.splines.length-1] = {};
                this.selected[this.splines.length-1][1] = ["anchorPoint"];
                this.mode = VectorCursorMode.MOVE;
            } else if (e.shiftKey && e.ctrlKey) {
                this.mode = VectorCursorMode.ROTATE;
            } else {
                this.startSelected = deepClone(this.selected);
                this.mode = VectorCursorMode.SELECT;
            }
        } finally {
            if (this.mode !== VectorCursorMode.NONE) {
                this.startPosition = [x, y];
            }
            this.updateCanvas();
            this.changed();
        }
    }

    /**
     * The 'mouseup' handler
     */
    onNodeMouseUp(e) {
        this.mode = VectorCursorMode.NONE;
        this.startPosition = null;
        this.startSelected = null;
        this.startRadians = null;
        this.updateCanvas();
        this.changed();
    }

    /**
     * Draws the select box and selects in two modes
     */
    rectangleSelect(points, addToSaved = false) {
        let [[x1, y1], [x2, y2]] = points,
            sx1 = Math.min(x1, x2),
            sx2 = Math.max(x1, x2),
            sy1 = Math.min(y1, y2),
            sy2 = Math.max(y1, y2);

        let drawable = new Drawable([
                new Point(sx1, sy1),
                new Point(sx2, sy1),
                new Point(sx2, sy2),
                new Point(sx1, sy2),
                new Point(sx1, sy1)
            ]),
            context = this.canvas.getContext("2d"),
            selected = {};

        if (addToSaved && !isEmpty(this.startSelected)) {
            selected = deepClone(this.startSelected);
        }

        for (let splineIndex in this.splines) {
            if (isEmpty(selected[splineIndex])) {
                selected[splineIndex] = {};
            }
            for (let pointIndex in this.splines[splineIndex].points) {
                let point = this.splines[splineIndex].points[pointIndex];
                if (isEmpty(selected[splineIndex][pointIndex])) {
                    selected[splineIndex][pointIndex] = [];
                }
                for (let pointName of ["anchorPoint", "controlPoint1", "controlPoint2"]) {
                    if (selected[splineIndex][pointIndex].indexOf(pointName) === -1 && 
                        !isEmpty(point[pointName]) &&
                        drawable.containsBounding(point[pointName])
                    ) {
                        selected[splineIndex][pointIndex].push(pointName);
                    }
                }
                if (isEmpty(selected[splineIndex][pointIndex])) {
                    delete selected[splineIndex][pointIndex];
                }
            }
            if (isEmpty(selected[splineIndex])) {
                delete selected[splineIndex];
            }
        }

        this.selected = selected;
        this.updateCanvas();

        context.setLineDash([4,2]);
        context.lineWidth = 2;
        context.strokeStyle = "#ffffff";
        drawable.stroke(context);
        context.setLineDash([]);
    }

    /**
     * Gets a list of selected points
     */
    getSelectedPoints() {
        let points = [];
        for (let selectedSplineIndex in this.selected) {
            for (let selectedPointIndex in this.selected[selectedSplineIndex]) {
                for (let selectedPointName of this.selected[selectedSplineIndex][selectedPointIndex]) {
                    if (!isEmpty(this.splines[selectedSplineIndex].points[selectedPointIndex][selectedPointName])) {
                        points.push(this.splines[selectedSplineIndex].points[selectedPointIndex][selectedPointName]);
                    }
                }
            }
        }
        return points;
    }

    /**
     * Copies any selected splines with an offset
     */
    copySelected(offset = 25) {
        let splineSelection = {};
        for (let splineIndex in this.selected) {
            let splineSelectedPoints = [];
            for (let pointIndex in this.selected[splineIndex]) {
                if (this.selected[splineIndex][pointIndex].indexOf("anchorPoint") !== -1) {
                    splineSelectedPoints.push(this.splines[splineIndex].points[pointIndex]);
                }
            }
            if (splineSelectedPoints.length >= 2) {
                let newPoints = splineSelectedPoints.map((point) => {
                        point = point.clone();
                        point.x = Math.min(point.x+offset, this.width);
                        point.y = Math.min(point.y+offset, this.height);
                        if (!isEmpty(point.controlPoint1)) {
                            point.cp1x = Math.min(point.cp1x+offset, this.width);
                            point.cp1y = Math.min(point.cp1y+offset, this.height);
                        }
                        if (!isEmpty(point.controlPoint2)) {
                            point.cp2x = Math.min(point.cp2x+offset, this.width);
                            point.cp2y = Math.min(point.cp2y+offset, this.height);
                        }
                        return point;
                    }),
                    newSpline = new EditableSplineVector(newPoints);
                this.splines.push(newSpline);
                splineSelection[this.splines.length-1] = {};
                for (let pointIndex in newSpline.points) {
                    splineSelection[this.splines.length-1][pointIndex] = [];
                    for (let pointName of ["anchorPoint", "controlPoint1", "controlPoint2"]) {
                        if (!isEmpty(newSpline.points[pointIndex][pointName])) {
                            splineSelection[this.splines.length-1][pointIndex].push(pointName);
                        }
                    }
                }
            }
        }
        if (!isEmpty(splineSelection)) {
            this.selected = splineSelection;
        }
        this.updateCanvas();
        this.changed();
    }

    /**
     * Deletes any selected splines
     */
    deleteSelected() {
        for (let splineIndex in this.selected) {
            for (let pointIndex in this.selected[splineIndex]) {
                if (this.selected[splineIndex][pointIndex].indexOf("anchorPoint") !== -1) {
                    this.splines[splineIndex].points[pointIndex] = null;
                }
            }
            this.splines[splineIndex].points = this.splines[splineIndex].points.filter((point) => point !== null);
            if (this.splines[splineIndex].points.length < 2) {
                this.splines[splineIndex] = null;
            }
        }

        this.splines = this.splines.filter((spline) => spline !== null);
        this.selected = {};
        this.updateCanvas();
        this.changed();
    }

    /**
     * The 'mousemove' handler
     */
    onNodeMouseMove(e) {
        let [eventX, eventY] = this.getCoordinates(e);
        switch (this.mode) {
            case VectorCursorMode.MOVE:
                let [lastX, lastY] = this.startPosition,
                    [deltaX, deltaY] = [lastX - eventX, lastY - eventY];
                for (let point of this.getSelectedPoints()) {
                    point.x -= deltaX;
                    point.y -= deltaY;
                }
                this.startPosition = [eventX, eventY];
                this.updateCanvas();
                break;
            case VectorCursorMode.ROTATE:
                let drawable = new Drawable(this.getSelectedPoints()),
                    thisPoint = new Point(eventX, eventY),
                    lastPoint = new Point(...this.startPosition),
                    thisVector = new Vector(thisPoint, drawable.center),
                    lastVector = new Vector(lastPoint, drawable.center),
                    rotation = thisVector.radians - lastVector.radians;
                drawable.rotate(rotation);
                this.updateCanvas();
                this.startPosition = [eventX, eventY];
                break;
            case VectorCursorMode.SELECT:
                this.rectangleSelect([this.startPosition, [eventX, eventY]], e.shiftKey);
                break;
            case VectorCursorMode.NONE:
                if (!isEmpty(this.getPointFromCoordinates(eventX, eventY))) {
                    this.node.css("cursor", "pointer");
                } else if (!isEmpty(this.getSplineFromCoordinates(eventX, eventY))) {
                    this.node.css("cursor", "cell");
                } else {
                    this.node.css("cursor", "default");
                }
                break;
        }
    }

    /**
     * The 'dblclick' handler
     */
    onDblClick(e) {
        let [eventX, eventY] = this.getCoordinates(e),
            point = this.getPointFromCoordinates(eventX, eventY);

        if (!isEmpty(point)) {
            let [activeSpline, activePoint] = point,
                splinePoint = this.splines[activeSpline].points[activePoint];

            if (splinePoint.pointType === SplinePoint.TYPE_LINEAR) {
                if (activePoint === 0) {
                    let nextPointSection = new Vector(...this.splines[activeSpline].points.slice(0, 2));
                    splinePoint.controlPoint2 = nextPointSection.halfway;
                } else if (activePoint === this.splines[activeSpline].points.length - 1) {
                    let previousPointSection = new Vector(...this.splines[activeSpline].points.slice(activePoint-1, activePoint+1));
                    splinePoint.controlPoint1 = previousPointSection.halfway;
                } else {
                    let nextPointSection = new Vector(...this.splines[activeSpline].points.slice(activePoint, activePoint+2)),
                        previousPointSection = new Vector(...this.splines[activeSpline].points.slice(activePoint-1, activePoint+1));
                    splinePoint.controlPoint1 = previousPointSection.halfway;
                    splinePoint.controlPoint2 = nextPointSection.halfway;
                }
                splinePoint.pointType = SplinePoint.TYPE_BEZIER;
            } else {
                splinePoint.pointType = SplinePoint.TYPE_LINEAR;
                delete splinePoint.controlPoint1;
                delete splinePoint.controlPoint2;
            }
            this.updateCanvas();
            this.changed();
        }
    }

    /**
     * On build, append canvas.
     */
    async build() {
        let node = await super.build();
        node.append(this.canvas);
        node.on("dblclick", (e) => this.onDblClick(e));
        node.on("mousemove", (e) => this.onNodeMouseMove(e), true);
        node.on("mousedown", (e) => this.onNodeMouseDown(e));
        node.on("mouseup", (e) => this.onNodeMouseUp(e));
        node.on("touchstart", (e) => this.onNodeMouseDown(e, true));
        node.on("touchmove", (e) => this.onNodeMouseMove(e), true);
        node.on("touchend", (e) => this.onNodeMouseUp(e));
        this.updateCanvas();
        return node;
    }
}

export { VectorView };
