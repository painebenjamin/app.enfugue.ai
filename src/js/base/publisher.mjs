/** @module base/publisher */
import { isEmpty } from "./helpers.mjs";

class Publisher {
    /**
     * The constructor initializes an empty object for storing subscriptions.
     */
    constructor() {
        this.subscriptions = {};
    }

    /**
     * This allows a class to subscribe to an event and pass callback(s).
     *
     * @param string $eventName The event name as a string. Taken verbatim.
     * @param callable $callback The function to call when an event is published.
     */
    subscribe(eventName, callback) {
        if (isEmpty(this.subscriptions[eventName])) {
            this.subscriptions[eventName] = [];
        }
        this.subscriptions[eventName].push(callback);
    }

    /**
     * This allows a class to remove a callback from subscriptions.
     * It's important to remember that this must test if two functions are equal
     * to each other. They are only equal if they are the exact same variable;
     * if it is the same function only defined somewhere else, it will not be
     * found.
     *
     * @param string $eventName The event name as a string. Taken verbatim.
     * @param callable $callback The function to remove.
     * @return bool True if the callback was found and removed.
     */
    unsubscribe(eventName, callback) {
        if (!isEmpty(this.subscriptions[eventName])) {
            for (let i in this.subscriptions[eventName]) {
                if (this.subscriptions[eventName][i] == callback) {
                    this.subscriptions[eventName] = this.subscriptions[eventName].slice(0, i).concat(this.subscriptions[eventName].slice(i+1));
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * This publishes an event, triggering any callbacks.
     * 
     * @param string $eventName The event name as a string. Taken verbatim.
     * @param mixed $payload A payload to pass to the callbacks. Optional.
     */
    async publish(eventName, payload = null) {
        if (isEmpty(this.subscriptions[eventName])) {
            return;
        }
        for (let callback of this.subscriptions[eventName]) {
            await callback(payload);
        }
    }
}

export { Publisher };
