import { Point, Drawable, Vector, CardinalDirection } from './geometry.mjs';

class PathFinder {
    /* Adapted from Qiaoyu & Yinrog, 2008
     * https://journals.sagepub.com/doi/pdf/10.1260/174830108788251809
     *
     * Finds a path between two points in a two-dimensional plane with
     * obstacle avoidance along 8 axes.
     */

    static stepSizes = [25, 5, 1];
    static maximumIterations = 1000;
    static obstacleBerth = 1;

    constructor(obstacles = []) {
        this.obstacles = obstacles;
    }

    find(start, end) {
        let stepIndex = 0,
            stepSize = this.constructor.stepSizes[stepIndex],
            berth = this.constructor.obstacleBerth,
            current = start.clone(),
            target = end.clone(),
            tracer = current.clone(),
            path = [start],
            vector = new Vector(current, end),
            direction = vector.estimateCardinal(stepSize),
            endReached = false,
            totalIterations = 0,
            obstaclesEncountered = {};

        while (!endReached) {
            let targetReached = false;
            while (!targetReached) {
                tracer.copy(current);
                switch (direction) {
                    case CardinalDirection.NORTH:
                        tracer.y -= stepSize;
                        targetReached = tracer.y < target.y;
                        break;
                    case CardinalDirection.NORTHEAST:
                        tracer.y -= stepSize;
                        tracer.x += stepSize;
                        targetReached =
                            tracer.y < target.y || tracer.x > target.x;
                        break;
                    case CardinalDirection.EAST:
                        tracer.x += stepSize;
                        targetReached = tracer.x > target.x;
                        break;
                    case CardinalDirection.SOUTHEAST:
                        tracer.x += stepSize;
                        tracer.y += stepSize;
                        targetReached =
                            tracer.y > target.y || tracer.x > target.x;
                        break;
                    case CardinalDirection.SOUTH:
                        tracer.y += stepSize;
                        targetReached = tracer.y > target.y;
                        break;
                    case CardinalDirection.SOUTHWEST:
                        tracer.x -= stepSize;
                        tracer.y += stepSize;
                        targetReached =
                            tracer.x < target.x || tracer.y > target.y;
                        break;
                    case CardinalDirection.WEST:
                        tracer.x -= stepSize;
                        targetReached = tracer.x < target.x;
                        break;
                    case CardinalDirection.NORTHWEST:
                        tracer.y -= stepSize;
                        tracer.x -= stepSize;
                        targetReached =
                            tracer.y < target.y || tracer.x < target.x;
                        break;
                }

                if (targetReached) {
                    // Step hasn't reached full granularity, decrease step size
                    if (stepIndex < this.constructor.stepSizes.length - 1) {
                        targetReached = false;
                        stepSize = this.constructor.stepSizes[++stepIndex];
                    } else {
                    }
                } else {
                    let hitObstacle = null;
                    for (let i = 0; i < this.obstacles.length; i++) {
                        if (
                            obstaclesEncountered[i] !== undefined &&
                            obstaclesEncountered[i].length >= 3
                        ) {
                            // Tried to go around an object too many times,
                            // it's likely the end point is inside the object itself.
                            continue;
                        }
                        if (this.obstacles[i].containsBounding(tracer, false)) {
                            hitObstacle = i;
                            break;
                        }
                    }

                    if (hitObstacle !== null) {
                        // If we hit an obstacle, find the nearest extreme in that obstacle
                        // and set that as the target. If we've already gone to this extreme,
                        // continuing navigating around the object.
                        let extremes = this.obstacles[hitObstacle].extremes,
                            lastPoint = path[path.length - 1],
                            nearestExtremeIndex = -1,
                            nearestExtremeDistance = null,
                            obstacleEncountered =
                                obstaclesEncountered[hitObstacle] !== undefined, // if true, this obstacle has been encountered before during pathing
                            compareTo = obstacleEncountered
                                ? target
                                : lastPoint; // If this obstacle was encountered before,

                        for (let i = 0; i < extremes.length; i++) {
                            if (
                                obstacleEncountered &&
                                obstaclesEncountered[hitObstacle].indexOf(i) !==
                                    -1
                            ) {
                                // Continue going around.
                                continue;
                            }

                            let extremePoint = extremes[i],
                                extremeDistance =
                                    compareTo.distanceTo(extremePoint);

                            if (
                                nearestExtremeDistance === null ||
                                extremeDistance < nearestExtremeDistance
                            ) {
                                nearestExtremeIndex = i;
                                nearestExtremeDistance = extremeDistance;
                            }
                        }
                        // Memoize the encountered obstacles
                        if (obstaclesEncountered[hitObstacle] === undefined) {
                            obstaclesEncountered[hitObstacle] = [];
                        }
                        obstaclesEncountered[hitObstacle].push(
                            nearestExtremeIndex
                        );

                        let nearestExtreme = extremes[nearestExtremeIndex];
                        target.copy(nearestExtreme);
                        current.copy(lastPoint);
                        direction = new Vector(
                            current,
                            target
                        ).estimateCardinal(stepSize);
                    } else {
                        // If we didn't hit an obstacle, keep going with tracer.
                        current.copy(tracer);
                    }
                }

                if (
                    !targetReached &&
                    totalIterations++ >= this.constructor.maximumIterations
                ) {
                    throw 'Maximum iterations reached.';
                }
            }
            endReached = current.equals(end, stepSize * 2);
            if (!endReached) {
                // Hit the target on some axis, but not the end. Reset granularity and push path parts.
                stepIndex = 0;
                stepSize = this.constructor.stepSizes[0];

                path.push(current.clone());

                if (current.equals(target)) {
                    // Hit the target on all axes. Set the target back to the end.
                    target.copy(end);
                    vector.start.copy(current);
                    direction = vector.estimateCardinal(stepSize);
                } else {
                    // Hit the target on one axis. Keep the target as is and change direction.
                    direction = new Vector(current, target).estimateCardinal(1);
                }
            }

            if (
                !endReached &&
                totalIterations++ >= this.constructor.maximumIterations
            ) {
                throw 'Maximum iterations reached.';
            }
        }
        path.push(end);
        return path;
    }
}

export { PathFinder };
