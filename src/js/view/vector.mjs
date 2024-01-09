/** @module view/vector */
import { isEmpty } from '../base/helpers.mjs';
import { View } from "./base.mjs";
import { Point, Drawable, Vector } from "../graphics/geometry.mjs";
import { Spline, SplinePoint } from "../graphics/spline.mjs";
import { ElementBuilder } from '../base/builder.mjs';

const E = new ElementBuilder();

class EditableSplineVector extends Spline {
    /**
     * @var int Half handle width
     */
    static handleWidth = 8;

    /**
     * @var int Half arrow length
     */
    static arrowLength = 10;

    /**
     * Gets the handle drawable about a point
     */
    getHandleAboutPoint(point) {
        return new Drawable([
            new Point(
                point.x-this.constructor.handleWidth,
                point.y-this.constructor.handleWidth
            ),
            new Point(
                point.x+this.constructor.handleWidth,
                point.y-this.constructor.handleWidth
            ),
            new Point(
                point.x+this.constructor.handleWidth,
                point.y+this.constructor.handleWidth
            ),
            new Point(
                point.x-this.constructor.handleWidth,
                point.y+this.constructor.handleWidth
            ),
            new Point(
                point.x-this.constructor.handleWidth,
                point.y-this.constructor.handleWidth
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
     * Draws the spline and points
     */
    draw(context) {
        context.lineWidth = 2;
        this.stroke(context);
        context.lineWidth = 1;
        for (let point of this.points) {
            let handle = this.getHandleAboutPoint(point);
            handle.stroke(context);
            if (point.pointType === SplinePoint.TYPE_BEZIER) {
                if (!isEmpty(point.controlPoint1)) {
                    let controlHandle = this.getHandleAboutPoint(point.controlPoint1),
                        controlLine = new Drawable([point.anchorPoint, point.controlPoint1]);
                    context.setLineDash([]);
                    controlHandle.stroke(context);
                    context.setLineDash([4,2]);
                    controlLine.stroke(context);
                }
                if (!isEmpty(point.controlPoint2)) {
                    let controlHandle = this.getHandleAboutPoint(point.controlPoint2),
                        controlLine = new Drawable([point.anchorPoint, point.controlPoint2]);
                    context.setLineDash([]);
                    controlHandle.stroke(context);
                    context.setLineDash([4,2]);
                    controlLine.stroke(context);
                }
            }
            context.setLineDash([]);
        }

        let [penultimate, end] = this.points.slice(-2);

        if (penultimate.pointType === SplinePoint.TYPE_LINEAR && end.pointType === SplinePoint.TYPE_BEZIER) {
            penultimate = this.quadraticBezierCurvePoint(penultimate, end.controlPoint1, end, 0.9);
        } else if (penultimate.pointType === SplinePoint.TYPE_BEZIER && end.pointType === SplinePoint.TYPE_LINEAR) {
            penultimate = this.quadraticBezierCurvePoint(penultimate, penultimate.controlPoint2, end, 0.9);
        } else if (penultimate.pointType !== SplinePoint.TYPE_LINEAR && end.pointType !== SplinePoint.TYPE_LINEAR) {
            penultimate = this.cubicBezierCurvePoint(penultimate, penultimate.controlPoint2, end.controlPoint1, end, 0.9);
        }

        let lastSection = new Vector(penultimate, end),
            arrowhead = new Drawable([
                new Point(
                    lastSection.end.x,
                    lastSection.end.y
                ),
                new Point(
                    lastSection.end.x - this.constructor.arrowLength,
                    lastSection.end.y - this.constructor.arrowLength
                ),
                new Point(
                    lastSection.end.x + this.constructor.arrowLength,
                    lastSection.end.y - this.constructor.arrowLength
                ),
                new Point(
                    lastSection.end.x,
                    lastSection.end.y
                )
            ]);

        arrowhead.rotateAbout(lastSection.rad + 3*Math.PI/2, lastSection.end);
        arrowhead.fill(context);
    }
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
    }

    /**
     * Gets the encoded data
     */
    get value() {

    }

    /**
     * Sets the encoded data
     */
    set value(newValue) {

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
    changed() {
        let callbackValue = this.value;
        for (let handler of this.onChangeCallbacks) {
            handler(callbackValue);
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
        this.updateCanvas();
        this.changed();
    }

    /**
     * Draws the canvas
     */
    updateCanvas() {
        let context = this.canvas.getContext("2d");
        context.clearRect(0, 0, this.width, this.height);
        context.strokeStyle = "#ffffff";
        context.fillStyle = "#ffffff";
        for (let spline of this.splines) {
            spline.draw(context);
        }
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
                if (point.x - EditableSplineVector.handleWidth <= x &&
                    point.x + EditableSplineVector.handleWidth >  x &&
                    point.y - EditableSplineVector.handleWidth <= y &&
                    point.y + EditableSplineVector.handleWidth >  y
                ) {
                    return [parseInt(i), parseInt(j), 0];
                }
                if (point.pointType === SplinePoint.TYPE_BEZIER) {
                    if (!isEmpty(point.controlPoint1)) {
                        if (point.controlPoint1.x - EditableSplineVector.handleWidth <= x &&
                            point.controlPoint1.x + EditableSplineVector.handleWidth >  x &&
                            point.controlPoint1.y - EditableSplineVector.handleWidth <= y &&
                            point.controlPoint1.y + EditableSplineVector.handleWidth >  y
                        ) {
                            return [parseInt(i), parseInt(j), 1];
                        }
                    }
                    if (!isEmpty(point.controlPoint2)) {
                        if (point.controlPoint2.x - EditableSplineVector.handleWidth <= x &&
                            point.controlPoint2.x + EditableSplineVector.handleWidth >  x &&
                            point.controlPoint2.y - EditableSplineVector.handleWidth <= y &&
                            point.controlPoint2.y + EditableSplineVector.handleWidth >  y
                        ) {
                            return [parseInt(i), parseInt(j), 2];
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
                console.log(splineIndex);
                return [parseInt(i), splineIndex+1];
            }
        }
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
        this.updateCanvas();
        this.lastX = null;
        this.lastY = null;
    }

    /**
     * The 'mousedown' handler
     */
    onNodeMouseDown(e) {
        if (e.type === "mousedown" && e.which !== 1) return;
        if (e.metaKey || e.ctrlKey) return;

        e.preventDefault();
        e.stopPropagation();

        let [x, y] = this.getCoordinates(e),
            point = this.getPointFromCoordinates(x, y);

        if (!isEmpty(point)) {
            if (e.altKey && point[2] === 0) {
                if (this.splines[point[0]].points.length === 2) {
                    this.splines = this.splines.slice(0, point[0]).concat(this.splines.slice(point[0]+1));
                } else {
                    this.splines[point[0]].removePoint(point[1]);
                }
            } else {
                [this.activeSpline, this.activePoint, this.activeControl] = point;
            }
            return;
        }

        let spline = this.getSplineFromCoordinates(x, y);
        if (!isEmpty(spline)) {
            this.splines[spline[0]].addPoint(spline[1]);
            return;
        }

        let newSpline = new EditableSplineVector([
            new SplinePoint(new Point(x, y), SplinePoint.TYPE_LINEAR),
            new SplinePoint(new Point(x, y), SplinePoint.TYPE_LINEAR)
        ]);
        this.splines.push(newSpline);
        this.activeSpline = this.splines.length - 1;
        this.activePoint = 1;
        this.activeControl = 0;
    }

    /**
     * The 'mouseup' handler
     */
    onNodeMouseUp(e) {
        this.activeSpline = null;
        this.activePoint = null;
        this.activeControl = null;
    }

    /**
     * The 'mousemove' handler
     */
    onNodeMouseMove(e) {
        let [eventX, eventY] = this.getCoordinates(e);
        if (
            !isEmpty(this.activeSpline) &&
            !isEmpty(this.activePoint) &&
            !isEmpty(this.activeControl)
        ) {
            let activePoint = this.splines[this.activeSpline].points[this.activePoint];
            switch (this.activeControl) {
                case 0:
                    activePoint = activePoint.anchorPoint;
                    break;
                case 1:
                    activePoint = activePoint.controlPoint1;
                    break;
                case 2:
                    activePoint = activePoint.controlPoint2;
                    break;
                default:
                    console.error("Bad active control", this.activeControl);
            }
            activePoint.x = eventX;
            activePoint.y = eventY;
        } else {
            if (!isEmpty(this.getPointFromCoordinates(eventX, eventY))) {
                this.node.css("cursor", "pointer");
            } else if (!isEmpty(this.getSplineFromCoordinates(eventX, eventY))) {
                this.node.css("cursor", "cell");
            } else {
                this.node.css("cursor", "default");
            }
        }
        this.updateCanvas();
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
        node.on("mouseenter", (e) => this.onNodeMouseEnter(e));
        node.on("mousemove", (e) => this.onNodeMouseMove(e));
        node.on("mousedown", (e) => this.onNodeMouseDown(e));
        node.on("mouseup", (e) => this.onNodeMouseUp(e));
        node.on("mouseleave", (e) => this.onNodeMouseLeave(e));
        node.on("touchstart", (e) => this.onNodeMouseDown(e, true));
        node.on("touchmove", (e) => this.onNodeMouseMove(e));
        node.on("touchend", (e) => this.onNodeMouseUp(e));
        this.updateCanvas();
        return node;
    }
}

export { VectorView };
