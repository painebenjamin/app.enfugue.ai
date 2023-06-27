const defaultConfiguration = {
    childList: true,
    subtree: true,
    attributes: true
};

class TrackedNode {
    constructor(node, tracker) {
        this.node = node;
        this.tracker = tracker;
        this.initialize();
    }

    update(node) {
        this.tracker.log('Executing update function.', this.node);
        this.tracker.updateFunction(this.node);
    }

    initialize() {
        this.tracker.log('Executing initialize function.', this.node);
        this.tracker.initializeFunction(this.node);
    }

    uninitialize() {
        this.tracker.log('Executing uninitialize function.', this.node);
        this.tracker.uninitializeFunction(this.node);
    }

    reinitialize() {
        this.uninitialize();
        this.initialize();
    }
}

class DOMWatcher {
    constructor(configuration) {
        configuration = configuration || {};

        this.name = configuration.name || 'Unnamed DOMWatcher';
        this.debug = configuration.debug;
        this.watchedNode = configuration.node || document.body;

        this.initializeFunction = configuration.initializeFunction || function(){};
        this.uninitializeFunction = configuration.uninitializeFunction || function(){};
        this.updateFunction = configuration.updateFunction || function(){};
        this.filterFunction = configuration.filterFunction || function(){ return true; };

        this.configuration = configuration.configuration || defaultConfiguration;

        this.trackedNodes = [];
0
        if (configuration.initialize !== false) {
            this.initialize();
        }

        this.observer = new MutationObserver(this.callback());
        this.observer.observe(this.watchedNode, this.configuration);
    }

    log(msg) {
        if (this.debug) {
            console.log.apply(
                null,
                [`DOMWatcher "${this.name}": ${msg}`].concat(
                    Array.from(arguments).slice(1)
                )
            );
        }
    }

    getTrackedNode(node) {
        return this.trackedNodes
            .filter((trackedNode) => trackedNode.node === node)
            .shift();
    }

    recurseNode(node) {
        let existingNode = this.getTrackedNode(node);

        if (existingNode) {
            existingNode.reinitialize();
        } else {
            if (this.filterFunction(node)) {
                this.addTrackedNode(node);
            }
            for (let child of node.childNodes) {
                if (child instanceof HTMLElement) {
                    this.recurseNode(child);
                }
            }
        }
    }

    updateTrackedNode(trackedNode, node) {
        this.log('Updating tracked node', trackedNode);
        trackedNode.update(node);
    }

    addTrackedNode(node) {
        this.log('Adding tracked node', node);
        this.trackedNodes.push(new TrackedNode(node, this));
    }

    removeTrackedNode(node) {
        this.log('Removing tracked node', node);
        let nodeIndex = this.trackedNodes
            .map((trackedNode) => trackedNode.node)
            .indexOf(node);

        this.trackedNodes[nodeIndex].uninitialize();
        this.trackedNodes = this.trackedNodes
            .slice(0, nodeIndex)
            .concat(this.trackedNodes.slice(nodeIndex + 1));
    }

    attributeChange(node) {
        let existingNode = this.getTrackedNode(node),
            filterNode = this.filterFunction(node);

        if (existingNode && !filterNode) {
            this.removeTrackedNode(node);
        } else if (!existingNode && filterNode) {
            this.addTrackedNode(node);
        } else if (existingNode && filterNode) {
            this.updateTrackedNode(existingNode, node);
        }
    }

    initialize() {
        this.recurseNode(this.watchedNode);
    }

    callback() {
        return (function (tracker) {
            return function (mutationsList, observer) {
                for (let mutation of mutationsList) {
                    switch (mutation.type) {
                        case 'childList':
                            for (let node of mutation.addedNodes) {
                                if (node instanceof HTMLElement) {
                                    tracker.recurseNode(node);
                                }
                            }
                            break;
                        case 'subtree':
                            // TODO
                            break;
                        case 'attributes':
                            tracker.attributeChange(mutation.target);
                            break;
                    }
                }
            };
        })(this);
    }
}

export { DOMWatcher };
