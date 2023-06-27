class MutexLock {
    constructor() {
        this.holder = Promise.resolve();
    }

    acquire() {
        let awaitResolve,
            temporaryPromise = new Promise((resolve) => {
                awaitResolve = () => resolve();
            }),
            returnValue = this.holder.then(() => awaitResolve);
        this.holder = temporaryPromise;
        return returnValue;
    }
}

class MutexScopeLock {
    constructor() {
        this.scopes = {};
    }

    acquire(name) {
        if (!this.scopes.hasOwnProperty(name)) {
            this.scopes[name] = new MutexLock();
        }
        return this.scopes[name].acquire();
    }
}

export { MutexLock, MutexScopeLock };
