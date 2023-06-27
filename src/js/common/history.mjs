/** @module model/database */
import { isEmpty, waitFor } from "../base/helpers.mjs";

/**
 * This database allows us to store a FIFO arbitrary page history
 */
class HistoryDatabase {
    /**
     * @var string The name of the IndexedDB
     */
    static databaseName = "enfugue";

    /**
     * @var int The version of the IndexedDB
     */
    static databaseVersion = 1;

    /**
     * When constructing the database, create the open request
     * @param int  $size The maximum number of history items to allow, after which the last item is popped.
     * @param bool $debug Whether or not to enable debug logging, default false
     */
    constructor(size, debug) {
        this.size = size;
        this.request = window.indexedDB.open(
            this.constructor.databaseName,
            this.constructor.databaseVersion
        );
        this.request.onerror = (e) => this.setError(e);
        this.request.onsuccess = (e) => this.setSuccess(e);
        this.request.onupgradeneeded = (e) => { this.setSuccess(e); this.migrate(); };
        this.debug = debug === true;
    }

    /**
     * Sets the error even when database opening fails
     * @param event $e The open event
     */
    setError(e) {
        this.error = e.target.errorCode;
    }

    /**
     * Sets the success event when database opening succeeds
     * @param event $e The open event
     */
    setSuccess(e) {
        this.database = e.target.result;
    }

    /**
     * Called when the database requires an 'upgrade', i.e. create the table.
     */
    migrate() {
        this.database.createObjectStore("history", {keyPath: "id"});
    }

    /**
     * Trims the database
     * Removes any history items that exceeed the configured size
     */
    trimDatabase() {
        return new Promise((resolve, reject) => {
            this.getIDs().then((idArray) => {
                let idsToRemove = idArray.slice(this.size + 1);
                if (idsToRemove.length > 0) {
                    if (this.debug) {
                        console.log("Removing history ID(s)", idsToRemove);
                    }
                    Promise.all(idsToRemove.map((id) => this.deleteByID(id)))
                        .then(resolve)
                        .catch(reject);
                } else if(this.debug) {
                    console.log(`Database does not need truncation (${idArray.length} < ${this.size+1})`);
                    resolve();
                }
            }).catch(reject);
        });
    }

    /**
     * Searches the database
     * Looks for any string in state that contains the value
     */
    searchHistoryItems(searchText) {
        return new Promise((resolve, reject) => {
            this.getHistoryItems().then((items) => {
                resolve(items.filter((item) => this.searchState(item.state, searchText)));
            });
        });
    }

    /**
     * Searches through a state part for text recursively.
     */
    searchState(state, searchText) {
        if (state === undefined || state === null) {
            return false;
        } else if (Array.isArray(state)) {
            for (let statePart of state) {
                if (this.searchState(statePart, searchText)) {
                    return true;
                }
            }
        } else if (typeof state === "object") {
            for (let propertyName of Object.getOwnPropertyNames(state)) {
                if (this.searchState(state[propertyName], searchText)) {
                    return true;
                }
            }
        } else if (typeof state !== "function") {
            return `${state}`.toLowerCase().indexOf(searchText.toLowerCase()) !== -1;
        }
        return false;
    }

    /**
     * Gets the last <N> history items.
     */
    getLastHistoryItems(count = 10) {
        return new Promise((resolve, reject) => {
            this.getLastIDs(count + 1).then((idArray) => {
                Promise.all(idArray.slice(1).map((id) => this.getByID(id))).then(resolve).catch(reject);
            }).catch(reject);
        });
    }

    /**
     * Gets all history items.
     */
    getHistoryItems() {
        return this.getLastHistoryItems(Infinity);
    }

    /**
     * Gets the last <n> IDs inserted into the database.
     */
    getLastIDs(count = 10) {
        return new Promise((resolve, reject) => {
            this.getDatabase().then((database) => {
                const transaction = database.transaction("history", "readonly");
                const history = transaction.objectStore("history");
                const request = history.openKeyCursor(null, "prev");
                let results = [], resultCount = 0;

                request.onsuccess = (e) => {
                    const cursor = e.target.result;
                    if (cursor) {
                        results.push(cursor.key);
                        if (++resultCount >= count) {
                            resolve(results);
                            transaction.abort();
                            return;
                        }
                        cursor.continue();
                    } else {
                        resolve(results); // Completed iteration
                        return;
                    }
                };
                request.onerror = (e) => {
                    reject(`Cursor request failed: ${e.errorCode}`);
                };
            }).catch(reject);
        });
    }

    /**
     * Gets all IDs inserted into the database.
     */
    getIDs() {
        return this.getLastIDs(Infinity);
    }

    /**
     * Gets only the last ID inserted into the database.
     */
    async getLastID() {
        let lastIDArray = await this.getLastIDs(1);
        if (lastIDArray.length === 0) return null;
        return lastIDArray[0];
    }

    /**
     * Gets a history item by ID
     */
    getByID(id) {
        return new Promise((resolve, reject) => {
            this.getDatabase().then((database) => {
                const history = database
                    .transaction(["history"], "readonly")
                    .objectStore("history");
                const request = history.get(id);
                request.onsuccess = (e) => resolve(e.target.result);
                request.onerror = (e) => {
                    reject(`Retrieving ${id} failed: ${e.errorCode}`);
                };
            }).catch(reject);
        });
    }

    /**
     * Deletes a history item by ID
     */
    deleteByID(id) {
        return new Promise((resolve, reject) => {
            this.getDatabase().then((database) => {
                const request = database
                    .transaction(["history"], "readwrite")
                    .objectStore("history")
                    .delete(id);
                request.onsuccess = (e) => {
                    if (this.debug) {
                        console.log("Successfully deleted history at ID", id);
                    }
                    resolve();
                };
                request.onerror = (e) => {
                    reject(`Couldn't delete ${id}: ${e.errorCode}`);
                };
            }).catch(reject);
        });
    }

    /**
     * Inserts a new record, effectively freezing the previous
     */
    flush(newState) {
        return new Promise((resolve, reject) => {
            this.getLastID().then((id) => {
                if (isEmpty(id)) {
                    reject("No current history stored, call setCurrentState at least once before flushing.");
                    return;
                }
                this.getDatabase().then((database) => {
                    const history = database
                        .transaction(["history"], "readwrite")
                        .objectStore("history");
                    const request = history.add({
                        "state": newState,
                        "timestamp": (new Date()).getTime(),
                        "id": id + 1
                    });
                    request.onsuccess = (e) => {
                        if (this.debug) {
                            console.log("Successfully inserted new history state.");
                        }
                        resolve(id + 1);
                        this.trimDatabase(); // Clean up database
                    };
                    request.onerror = (e) => {
                        reject(`Inserting new history failed: ${e.errorCode}`);
                    };
                }).catch(reject);
            });
        });
    }

    /**
     * Sets the current history state.
     * @return Promise
     */
    setCurrentState(state) {
        return new Promise((resolve, reject) => {
            this.getLastID().then((id) => {
                this.getDatabase().then((database) => {
                    const history = database
                        .transaction(["history"], "readwrite")
                        .objectStore("history");
                    if (id === null) {
                        const request = history.add({
                            "state": state, 
                            "id": 1,
                            "timestamp": (new Date()).getTime()
                        });
                        request.onsuccess = (e) => {
                            if (this.debug) {
                                console.log("Successfully inserted first history state.");
                            }
                            resolve(1);
                        };
                        request.onerror = (e) => {
                            reject(`Inserting first history failed: ${e.errorCode}`);
                        };
                    } else {
                        this.getByID(id).then((datum) => {
                            datum.state = state;
                            datum.timestamp = (new Date()).getTime();
                            const history = database
                                .transaction(["history"], "readwrite")
                                .objectStore("history");
                            const updateRequest = history.put(datum);
                            updateRequest.onsuccess = (e) => {
                                if (this.debug) {
                                    console.log(`Successfully updated current state by ID ${id}`);
                                }
                                resolve(id);
                            };
                            updateRequest.onerror = (e) => {
                                reject(`Updating ${id} failed: ${e.errorCode}`);
                            };
                        }).catch(reject);
                    }
                }).catch(reject);
            }).catch(reject);
        });
    }

    /**
     * Gets the current state.
     * @return Promise
     */
    getCurrentState() {
        return new Promise((resolve, reject) => {
            this.getLastID().then((id) => {
                if (id === null) {
                    if (this.debug) {
                        console.log("No history currently saved, returning null.");
                    }
                    resolve(null);
                    return;
                }
                this.getDatabase().then((database) => {
                    const request = database
                        .transaction(["history"], "readonly")
                        .objectStore("history").get(id);
                    request.onsuccess = (e) => {
                        if (this.debug) {
                            console.log(`Successfully retrieved current state by ID ${id}.`);
                        }
                        resolve(request.result.state);
                    };
                    request.onerror = (e) => {
                        reject(`Couldn't get history ${id}: ${e.errorCode}`);
                    };
                }).catch(reject);
            }).catch(reject);
        });
    }

    /**
     * Gets the database object.
     * @return Promise
     */
    async getDatabase() {
        await waitFor(() => !isEmpty(this.error) || !isEmpty(this.database));
        if (!isEmpty(this.error)) throw `Error: ${this.error}`;
        return this.database;
    }
};

export { HistoryDatabase };
