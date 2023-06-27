import { DOMWatcher } from '../base/watcher.mjs';
import { uuid } from '../base/helpers.mjs';

const eventTrackingConfig = {
    childList: true,
    subtree: true,
    attributes: true
};

let trackers = [];

class UserEventTracker extends DOMWatcher {
    constructor(configuration) {
        super(configuration);
        this.filterFunction = this.nodeFilter;
        this.initializeFunction = this.nodeInitialize;
        this.uninitializeFunction = this.nodeUninitialize;
        this.eventTrackerCallbacks = configuration.eventTrackerCallbacks;
        this.boundEvents = {};
    }

    nodeFilter(node) {
        return this.getTrackingProperties(node) !== undefined;
    }

    nodeInitialize(node) {
        let trackingProperties = this.getTrackingProperties(node);

        this.boundEvents[trackingProperties.id] = {};

        for (let eventName of trackingProperties.events) {
            let callbackFunction = (function (
                trackedNode,
                trackingProperties,
                callbacks
            ) {
                return function (e) {
                    let value = trackingProperties.value;
                    if (typeof value === 'function') {
                        value = value();
                    }
                    for (let callback of callbacks) {
                        callback(e, trackingProperties.category, value);
                    }
                };
            })(node, trackingProperties, this.eventTrackerCallbacks);
            this.boundEvents[trackingProperties.id][eventName] =
                callbackFunction;
            node.addEventListener(eventName, callbackFunction);
        }
    }

    nodeUninitialize(node) {
        let nodeId = node.getAttribute('data-tracking-id');

        for (let eventName in this.boundEvents[nodeId]) {
            node.removeEventListener(
                eventName,
                this.boundEvents[nodeId][eventName]
            );
        }

        delete this.boundEvents[nodeId];
    }

    addTrackedNode(node) {
        node.setAttribute('data-tracking-id', uuid());
        super.addTrackedNode(node);
    }

    getTrackingProperties(node) {
        try {
            let events = node.getAttribute('data-tracking-events'),
                id = node.getAttribute('data-tracking-id');

            if (events !== undefined && events !== null) {
                let valueType = node.getAttribute('data-tracking-value-type');
                if (valueType === null) {
                    valueType = 'static';
                }

                let value = node.getAttribute('data-tracking-value');

                if (valueType === 'function') {
                    value = (function (node, value) {
                        return function () {
                            return function (value) {
                                return eval(value);
                            }.call(node, value);
                        };
                    })(node, value);
                }

                return {
                    id: id,
                    events: events.split(','),
                    category: node.getAttribute('data-tracking-category'),
                    value: value
                };
            }
        } catch (e) {}
    }
}

let createTracker = function (node, debug) {
    let configuration = {
        node: node,
        eventTrackerCallbacks: Array.from(arguments).slice(2),
        debug: debug
    };

    let newTracker = new UserEventTracker(configuration);
    trackers.push(newTracker);

    return newTracker;
};

export let EventTracking = {
    track: createTracker
};
