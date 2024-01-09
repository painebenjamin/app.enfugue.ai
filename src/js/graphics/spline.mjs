/** @module graphics/spline */
import { Point, Drawable } from './geometry.mjs';
import { roundTo, shiftingFrameIterator } from '../base/helpers.mjs';

/**
 * Represents a point on a spline, with option control points for bezier
 */
class SplinePoint {
    /**
     * @var int constant for linear type
     */
    static TYPE_LINEAR = 0;

    /**
     * @var int constant for bezier type
     */
    static TYPE_BEZIER = 1;

    /**
     * Construct requires anchor and type
     */
    constructor(anchorPoint, pointType, controlPoint1, controlPoint2) {
        this.anchorPoint = anchorPoint;
        this.pointType = pointType;
        this.controlPoint1 = controlPoint1;
        this.controlPoint2 = controlPoint2;
    }

    /**
     * Create a clone of this point
     */
    clone() {
        return new SplinePoint(
            this.anchorPoint.clone(),
            this.pointType,
            this.controlPoint1 === undefined
                ? undefined
                : this.controlPoint1.clone(),
            this.controlPoint2 === undefined
                ? undefined
                : this.controlPoint2.clone()
        );
    }

    /**
     * Make this point copy another
     */
    copy(otherPoint) {
        if (otherPoint instanceof SplinePoint) {
            this.anchorPoint.x = otherPoint.x;
            this.anchorPoint.y = otherPoint.y;
            this.pointType = otherPoint.pointType;

            if (otherPoint.controlPoint1 === undefined) {
                this.controlPoint1 = undefined;
            } else if (this.controlPoint1 === undefined) {
                this.controlPoint1 = otherPoint.controlPoint1.clone();
            } else {
                this.controlPoint1.copy(otherPoint.controlPoint1);
            }

            if (otherPoint.controlPoint2 === undefined) {
                this.controlPoint2 = undefined;
            } else if (this.controlPoint2 === undefined) {
                this.controlPoint2 = otherPoint.controlPoint2.clone();
            } else {
                this.controlPoint2.copy(otherPoint.controlPoint2);
            }
        } else if (otherPoint instanceof Point) {
            this.anchorPoint.x = otherPoint.x;
            this.anchorPoint.y = otherPoint.y;
        } else {
            throw 'Cannot copy point of type ' + typeof otherPoint;
        }
    }

    /**
     * @param int anchor point X
     */
    set x(x) {
        this.anchorPoint.x = x;
    }

    /**
     * @return int anchor point X
     */
    get x() {
        return this.anchorPoint.x;
    }

    /**
     * @param int anchor point Y
     */
    set y(y) {
        this.anchorPoint.y = y;
    }

    /**
     * @return int anchor point Y
     */
    get y() {
        return this.anchorPoint.y;
    }

    /**
     * @return int control point 1 X
     */
    get cp1x() {
        return this.controlPoint1.x;
    }

    /**
     * @param int control point 1 X
     */
    set cp1x(x) {
        this.controlPoint1.x = x;
    }

    /**
     * @return int control point 1 Y
     */
    get cp1y() {
        return this.controlPoint1.y;
    }

    /**
     * @param int control point 1 Y
     */
    set cp1y(y) {
        this.controlPoint1.y = y;
    }

    /**
     * @return int control point 2 x
     */
    get cp2x() {
        return this.controlPoint2.x;
    }

    /**
     * @param int control point 2 x
     */
    set cp2x(x) {
        this.controlPoint2.x = x;
    }

    /**
     * @return int control point 2 y
     */
    get cp2y() {
        return this.controlPoint2.y;
    }

    /**
     * @param int control point 2 y
     */
    set cp2y(y) {
        this.controlPoint2.y = y;
    }
}

/**
 * Spline collects many SplinePoints
 */
class Spline extends Drawable {
    /**
     * Given a point and tolerance, determine if the passed point
     * is along this spline. If it is, return the index of the point
     * before where the passed point would be inserted were it to be
     * added to this spline.
     */
    pointAlongSpline(point, tolerance) {
        let startIndex = 0;
        for (let [startPoint, endPoint] of shiftingFrameIterator(this.points, 2)) {
            if (
                startPoint.pointType === SplinePoint.TYPE_LINEAR &&
                endPoint.pointType === SplinePoint.TYPE_LINEAR
            ) {
                if (
                    (point.x >= startPoint.x && point.x < endPoint.x) ||
                    (point.x < startPoint.x && point.x >= endPoint.x)
                ) {
                    let slope = (endPoint.y-startPoint.y)/(endPoint.x - startPoint.x),
                        yValue = slope * (point.x - startPoint.x) + startPoint.y;

                    if (
                        yValue - tolerance <= point.y &&
                        yValue + tolerance >= point.y
                    ) {
                        return startIndex;
                    }
                }
            } else if (
                startPoint.pointType === SplinePoint.TYPE_BEZIER &&
                endPoint.pointType === SplinePoint.TYPE_BEZIER
            ) {
                for (let i = 1; i <= tolerance * 3; i++) {
                    let t = i / (tolerance * 3),
                        p0 = startPoint,
                        p1 = startPoint.controlPoint2,
                        p2 = endPoint.controlPoint1,
                        p3 = endPoint,
                        Bx =
                            Math.pow(1 - t, 3) * p0.x +
                            3 * Math.pow(1 - t, 2) * t * p1.x +
                            3 * (1 - t) * Math.pow(t, 2) * p2.x +
                            Math.pow(t, 3) * p3.x,
                        By =
                            Math.pow(1 - t, 3) * p0.y +
                            3 * Math.pow(1 - t, 2) * t * p1.y +
                            3 * (1 - t) * Math.pow(t, 2) * p2.y +
                            Math.pow(t, 3) * p3.y;

                    if (
                        Bx - tolerance <= point.x &&
                        Bx + tolerance >= point.x &&
                        By - tolerance <= point.y &&
                        By + tolerance >= point.y
                    ) {
                        return startIndex;
                    }
                }
            } else {
                for (let i = 1; i <= tolerance * 2; i++) {
                    let t = i / (tolerance * 2),
                        p0 = startPoint,
                        p1 =
                            startPoint.pointType === SplinePoint.TYPE_BEZIER
                                ? startPoint.controlPoint2
                                : endPoint.controlPoint1,
                        p2 = endPoint,
                        Bx =
                            Math.pow(1 - t, 2) * p0.x +
                            2 * (1 - t) * t * p1.x +
                            Math.pow(t, 2) * p2.x,
                        By =
                            Math.pow(1 - t, 2) * p0.y +
                            2 * (1 - t) * t * p1.y +
                            Math.pow(t, 2) * p2.y;
                    if (
                        Bx - tolerance <= point.x &&
                        Bx + tolerance >= point.x &&
                        By - tolerance <= point.y &&
                        By + tolerance >= point.y
                    ) {
                        return startIndex;
                    }
                }
            }
            startIndex++;
        }
    }

    /**
     * Draw the path on the canvas
     */
    drawPath(context) {
        context.beginPath();
        context.moveTo(this.points[0].x, this.points[0].y);

        for (let [startPoint, endPoint] of shiftingFrameIterator(
            this.points,
            2
        )) {
            if (
                startPoint.pointType === SplinePoint.TYPE_LINEAR &&
                endPoint.pointType === SplinePoint.TYPE_LINEAR
            ) {
                context.lineTo(endPoint.x, endPoint.y);
            } else if (
                startPoint.pointType === SplinePoint.TYPE_LINEAR &&
                endPoint.pointType === SplinePoint.TYPE_BEZIER
            ) {
                context.quadraticCurveTo(
                    endPoint.cp1x,
                    endPoint.cp1y,
                    endPoint.x,
                    endPoint.y
                );
            } else if (
                startPoint.pointType === SplinePoint.TYPE_BEZIER &&
                endPoint.pointType === SplinePoint.TYPE_LINEAR
            ) {
                context.quadraticCurveTo(
                    startPoint.cp2x,
                    startPoint.cp2y,
                    endPoint.x,
                    endPoint.y
                );
            } else {
                context.bezierCurveTo(
                    startPoint.cp2x,
                    startPoint.cp2y,
                    endPoint.cp1x,
                    endPoint.cp1y,
                    endPoint.x,
                    endPoint.y
                );
            }
        }
    }
}

export { SplinePoint, Spline };
