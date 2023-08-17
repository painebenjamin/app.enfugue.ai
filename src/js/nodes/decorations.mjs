/** @module nodes/decorations */
import { isEmpty, deepClone, sleep } from '../base/helpers.mjs';
import { View } from '../view/base.mjs';
import {
    Point,
    Drawable,
    Vector,
    CardinalDirection
} from '../graphics/geometry.mjs';
import { PathFinder } from '../graphics/paths.mjs';
import { SplinePoint, Spline } from '../graphics/spline.mjs';

const splineHideAlpha = 0.25,
    splineShowAlpha = 0.5,
    frameDelay = 50,
    timedDrawDelay = 1000;

class NodeConnectionSpline extends Spline {
    static lineJoin = 'miter';
    constructor(points, width, shrink, color, exclusions) {
        super(points);
        this.width = width;
        this.shrink = shrink;
        this.rgb = color;
        this.a = splineHideAlpha;
        this.exclusions = exclusions;
    }

    set fixed(newFixed) {
        if (newFixed) {
            this.a = splineShowAlpha;
        } else {
            this.a = splineHideAlpha;
        }
    }

    get start() {
        return this.points[0];
    }

    get end() {
        return this.points[this.points.length - 1];
    }

    set start(newStartPoint) {
        this.start.x = Math.round(newStartPoint.x);
        this.start.y = Math.round(newStartPoint.y);
        this.recalculate();
    }

    set end(newEndPoint) {
        this.end.x = Math.round(newEndPoint.x);
        this.end.y = Math.round(newEndPoint.y);
        this.recalculate();
    }

    set rgb(newColor) {
        [this.r, this.g, this.b] = newColor;
    }

    exclude(newExclusionZones) {
        this.exclusions = newExclusionZones;
        this.recalculate();
    }

    recalculate() {
        // All splines are left-to-right
        try {
            let start = this.start.anchorPoint,
                end = this.end.anchorPoint,
                startPadding = start.clone().add(new Point(this.width, 0)),
                endPadding = end.clone().subtract(new Point(this.width, 0)),
                finder = new PathFinder(this.exclusions),
                path = finder.find(startPadding, endPadding),
                fullPath = [start, ...path, end];

            this.points = fullPath.map((point) => {
                return new SplinePoint(point, SplinePoint.TYPE_LINEAR);
            });
        } catch (e) {
            // Error in finding path. Leave as last path.
        }
        /*
    start.cp2y = start.y;
    start.cp2x = end.x;

    end.cp1x = start.x;
    end.cp1y = end.y;
    */
    }

    stroke(context) {
        this.drawPath(context);
        context.lineJoin = this.constructor.lineJoin;
        context.lineWidth = this.width - this.shrink;
        context.strokeStyle = `rgba(${this.r},${this.g},${this.b},${this.a})`;
        context.stroke();

        // Draw caps
        let start = this.points[0],
            end = this.points[this.points.length - 1],
            startTopCap = [
                new Point(start.x, start.y - this.shrink),
                new Point(start.x, start.y - this.shrink * 1.5),
                new Point(start.x + this.shrink / 2, start.y - this.shrink)
            ],
            startBottomCap = [
                new Point(start.x, start.y + this.shrink),
                new Point(start.x, start.y + this.shrink * 1.5),
                new Point(start.x + this.shrink / 2, start.y + this.shrink)
            ],
            endTopCap = [
                new Point(end.x, end.y - this.shrink),
                new Point(end.x, end.y - this.shrink * 1.5),
                new Point(end.x - this.shrink / 2, end.y - this.shrink)
            ],
            endBottomCap = [
                new Point(end.x, end.y + this.shrink),
                new Point(end.x, end.y + this.shrink * 1.5),
                new Point(end.x - this.shrink / 2, end.y + this.shrink)
            ];

        for (let [p1, p2, p3] of [
            startTopCap,
            startBottomCap,
            endTopCap,
            endBottomCap
        ]) {
            context.beginPath();
            context.moveTo(p1.x, p1.y);
            context.lineTo(p2.x, p2.y);
            //context.quadraticCurveTo(p1.x, p1.y, p3.x, p3.y);
            context.lineTo(p3.x, p3.y);
            context.fillStyle = `rgba(${this.r},${this.g},${this.b},${this.a})`;
            context.fill();
        }
    }
}

class NodeEditorDecorationsView extends View {
    static tagName = 'canvas';

    constructor(config, editor, width, height, lineWidth) {
        super(config);
        this.editor = editor;
        this.width = width;
        this.height = height;
        this.splines = [];
        this.startTimer();
    }

    get exclusions() {
        return this.editor.nodes.map((node) => node.drawable);
    }

    recalculate() {
        let exclusionZones = this.exclusions;
        for (let spline of this.splines) {
            spline.exclude(exclusionZones);
        }
    }

    draw() {
        if (this.node !== undefined) {
            if (this.node.element === undefined) {
                // Hasn't been rendered once yet.
                return;
            }
            let context = this.node.element.getContext('2d');
            context.clearRect(0, 0, this.width, this.height);
            for (let spline of this.splines) {
                spline.stroke(context);
            }
        }
    }

    startTimer() {
        let timerStarted;
        if (this.node !== undefined) {
            if (this.node.element !== undefined) {
                timerStarted = true;
                this.executeTimer();
            }
        }
        if (!timerStarted) {
            setTimeout(() => this.startTimer(), frameDelay);
        }
    }

    executeTimer() {
        if (
            this.node !== undefined &&
            this.node.element !== undefined &&
            document.documentElement.contains(this.node.element)
        ) {
            this.draw();
            setTimeout(() => this.executeTimer(), timedDrawDelay);
        }
    }

    add(point, height, shrink, color) {
        let splinePoint = new SplinePoint(point, SplinePoint.TYPE_LINEAR),
            spline = new NodeConnectionSpline(
                [splinePoint, splinePoint.clone()],
                height,
                shrink,
                color,
                this.exclusions
            );
        this.splines.push(spline);
        return spline;
    }

    remove(spline) {
        this.splines = this.splines.filter((s) => s != spline);
    }

    setDimension(width, height) {
        this.width = width;
        this.height = height;
        if (this.node !== undefined) {
            this.node
                .height(this.height)
                .width(this.width)
                .css({
                    height: `${this.height}px`,
                    width: `${this.width}px`
                });
        }
    }

    async build() {
        let node = await super.build();
        node.height(this.height)
            .width(this.width)
            .css({
                height: `${this.height}px`,
                width: `${this.width}px`
            });
        return node;
    }
}

export { NodeEditorDecorationsView, NodeConnectionSpline };
