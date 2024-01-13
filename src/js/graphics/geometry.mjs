import { roundTo } from '../base/helpers.mjs';

class Point {
    constructor(x, y) {
        if (x === undefined) x = 0;
        if (y === undefined) y = 0;
        this.x = x;
        this.y = y;
    }

    clone() {
        return new this.constructor(this.x, this.y);
    }

    equals(otherPoint, tolerance = 0) {
        return (
            this.x - tolerance <= otherPoint.x &&
            otherPoint.x <= this.x + tolerance &&
            this.y - tolerance <= otherPoint.y &&
            otherPoint.y <= this.y + tolerance
        );
    }

    copy(otherPoint) {
        this.x = otherPoint.x;
        this.y = otherPoint.y;
        return this;
    }

    subtract(otherPoint) {
        this.x -= otherPoint.x;
        this.y -= otherPoint.y;
        return this;
    }

    add(otherPoint) {
        this.x += otherPoint.x;
        this.y += otherPoint.y;
        return this;
    }

    scale(factor) {
        this.x *= factor;
        this.y *= factor;
    }

    distanceTo(otherPoint) {
        return Math.sqrt(
            Math.pow(otherPoint.x - this.x, 2) +
                Math.pow(otherPoint.y - this.y, 2)
        );
    }

    toString() {
        return `(${this.x},${this.y})`;
    }

    rotate(radians) {
        let x = this.x,
            y = this.y;

        this.x = x * Math.cos(radians) - y * Math.sin(radians);
        this.y = y * Math.cos(radians) + x * Math.sin(radians);
        return this;
    }
}

class Triplet {
    constructor(start, middle, end) {
        this.start = start;
        this.middle = middle;
        this.end = end;
    }

    orientation() {
        let orientationValue =
            (this.middle.y - this.start.y) * (this.end.x - this.middle.x) -
            (this.middle.x - this.start.x) * (this.end.y - this.middle.y);
        return orientation === 0 ? 0 : orientation > 0 ? 1 : 2;
    }
}

class CardinalDirection {
    static NORTH = 0;
    static NORTHEAST = 1;
    static EAST = 2;
    static SOUTHEAST = 3;
    static SOUTH = 4;
    static SOUTHWEST = 5;
    static WEST = 6;
    static NORTHWEST = 7;

    static toName(direction) {
        return [
            'North',
            'Northeast',
            'East',
            'Southeast',
            'South',
            'Southwest',
            'West',
            'Northwest'
        ][direction];
    }
}

class Vector {
    constructor(start, end) {
        this.start = start;
        this.end = end;
    }

    get radians() {
        return Math.atan2(
            this.end.y - this.start.y,
            this.end.x - this.start.x
        );
    }

    get actualCardinal() {
        let degrees = (this.radians * 180) / Math.PI + 180,
            nearestCardinalDegrees = roundTo(degrees, 45),
            nearestCardinalIndex = parseInt(nearestCardinalDegrees / 45),
            cardinalNumber = nearestCardinalIndex - 2;

        if (cardinalNumber < 0) {
            cardinalNumber += 7;
        }

        return cardinalNumber;
    }

    estimateCardinal(tolerance = 0) {
        let equalAlongX =
                this.start.x - tolerance <= this.end.x &&
                this.end.x <= this.start.x + tolerance,
            equalAlongY =
                this.start.y - tolerance <= this.end.y &&
                this.end.y <= this.start.y + tolerance;

        if (equalAlongY)
            return this.end.x > this.start.x
                ? CardinalDirection.EAST
                : CardinalDirection.WEST;
        if (equalAlongX)
            return this.end.y > this.start.y
                ? CardinalDirection.SOUTH
                : CardinalDirection.NORTH;
        if (this.end.x > this.start.x)
            return this.end.y > this.start.y
                ? CardinalDirection.SOUTHEAST
                : CardinalDirection.NORTHEAST;
        return this.end.y > this.start.y
            ? CardinalDirection.SOUTHWEST
            : CardinalDirection.NORTHWEST;
    }

    get deltaX() {
        return this.end.x - this.start.x;
    }

    get deltaY() {
        return this.end.y - this.start.y;
    }

    get slope() {
        return this.deltaY/this.deltaX;
    }

    get rad() {
        return Math.atan2(this.deltaY, this.deltaX);
    }

    get magnitude() {
        return this.start.distanceTo(this.end);
    }

    get halfway() {
        return new Point(
            (this.start.x + this.end.x) / 2,
            (this.start.y + this.end.y) / 2
        );
    }

    onVector(point) {
        return (
            point.x <= Math.max(this.start.x, this.end.x) &&
            point.x >= Math.max(this.start.x, this.end.x) &&
            point.y <= Math.max(this.start.y, this.end.y) &&
            point.y >= Math.min(this.start.y, this.end.y)
        );
    }

    intersects(otherVector) {
        let triplets = [
                new Triplet(this.start, this.end, otherVector.start),
                new Triplet(this.start, this.end, otherVector.end),
                new Triplet(otherVector.start, otherVector.end, this.start),
                new Triplet(otherVector.start, otherVector.end, this.end)
            ],
            orientations = triplets.map((triplet) => triplet.orientation());

        if (
            orientations[0] !== orientations[1] &&
            orientations[2] !== orientations[3]
        ) {
            return true;
        }

        if (orientations[0] === 0 && this.onVector(otherVector.start))
            return true;
        if (orientations[1] === 0 && this.onVector(otherVector.end))
            return true;
        if (orientations[2] === 0 && otherVector.onVector(this.start))
            return true;
        if (orientations[3] === 0 && otherVector.onVector(this.end))
            return true;
        return false;
    }
}

class Drawable {
    constructor(points, sort=false) {
        this.points = points;
        if (sort) {
            this.points.sort((a, b) => a.x - b.x);
        }
        this.center = points.slice(1).reduce((carry, point) => {
            carry.x += point.x;
            carry.y += point.y;
            return carry;
        }, this.points[0].clone());
        this.center.x /= this.points.length;
        this.center.y /= this.points.length;
        this.minimum = this.getMinimum();
        this.maximum = this.getMaximum();
    }

    clone() {
        return new this.constructor(
            [].concat(this.points.map((point) => point.clone()))
        );
    }

    getMaximum() {
        return new Point(
            Math.max(...this.points.map((p) => p.x)),
            Math.max(...this.points.map((p) => p.y))
        );
    }

    getMinimum() {
        return new Point(
            Math.min(...this.points.map((p) => p.x)),
            Math.min(...this.points.map((p) => p.y))
        );
    }

    translatePoint(otherPoint) {
        for (let point of this.points) {
            point.add(otherPoint);
        }
        this.center.add(otherPoint);
        return this;
    }

    translateX(deltaX) {
        for (let point of this.points) {
            point.x += deltaX;
        }
        this.center.x += deltaX;
        return this;
    }

    translateY(deltaY) {
        for (let point of this.points) {
            point.y += deltaY;
        }
        this.center.y += deltaY;
        return this;
    }

    get bounds() {
        let min = this.minimum,
            max = this.maximum;

        return [min.x, min.y, max.x - min.x, max.y - min.y];
    }

    get extremes() {
        let min = this.minimum,
            max = this.maximum;

        return [
            new Point(min.x, min.y),
            new Point(max.x, min.y),
            new Point(max.x, max.y),
            new Point(min.x, max.y)
        ];
    }

    containsBounding(otherPoint, includeEdge = true) {
        let min = this.minimum,
            max = this.maximum;

        if (includeEdge) {
            return (
                min.x <= otherPoint.x &&
                otherPoint.x <= max.x &&
                min.y <= otherPoint.y &&
                otherPoint.y <= max.y
            );
        } else {
            return (
                min.x < otherPoint.x &&
                otherPoint.x < max.x &&
                min.y < otherPoint.y &&
                otherPoint.y < max.y
            );
        }
    }

    contains(otherPoint) {
        let extreme = new Vector(otherPoint, new Point(Infinity, otherPoint.y)),
            count = 0,
            i = 0,
            next,
            vector;

        do {
            next = (i + 1) % this.length;
            vector = new Vector(this.points[i], this.points[next]);
            if (vector.intersects(extreme)) {
                if (
                    new Triplet(
                        this.points[i],
                        otherPoint,
                        this.points[next]
                    ).orientation() === 0
                ) {
                    return vector.onVector(otherPoint);
                }
            }
        } while (i !== 0);

        return count & 1;
    }

    translate(vector) {
        for (let point of this.points) {
            point.x += vector.deltaX;
            point.y += vector.deltaY;
        }
        this.center.x += vector.deltaX;
        this.center.y += vector.deltaY;
        return this;
    }

    rotate(radians) {
        return this.rotateAbout(radians, this.center);
    }

    rotateAbout(radians, center) {
        let normalizedPoint;
        for (let point of this.points) {
            normalizedPoint = point.clone().subtract(center);
            normalizedPoint.rotate(radians);
            normalizedPoint.add(center);
            point.copy(normalizedPoint);
        }
        return this;
    }

    scale(factor) {
        return this.scaleAbout(factor, this.center);
    }

    scaleAbout(factor, center) {
        let normalizedPoint;
        for (let point of this.points) {
            normalizedPoint = point.clone().subtract(center);
            normalizedPoint.scale(factor);
            normalizedPoint.add(center);
            point.copy(normalizedPoint);
        }
        return this;
    }

    drawPath(context) {
        context.beginPath();
        context.moveTo(this.points[0].x, this.points[0].y);
        for (let point of this.points.slice(1)) {
            context.lineTo(point.x, point.y);
        }
    }

    stroke(context) {
        this.drawPath(context);
        context.stroke();
    }

    fill(context) {
        this.drawPath(context);
        context.fill();
    }
}

export { Point, Vector, Drawable, CardinalDirection };
