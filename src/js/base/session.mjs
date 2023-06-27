import { isEmpty } from '../base/helpers.mjs';

class SessionStorage {
    constructor(driver) {
        this.driver = driver;
        this.scopes = {};
    }

    getScope(key) {
        if (!this.scopes.hasOwnProperty(key)) {
            this.scopes[key] = new ScopedStorage(key, this.driver);
        }
        return this.scopes[key];
    }

    getAll() {
        let storage = this;
        return Object.getOwnPropertyNames(this.scopes).reduce(function (
            acc,
            item
        ) {
            acc[item] = storage.getScope(item).getAll();
            return acc;
        },
        {});
    }

    clear() {
        for (let scope in this.scopes) {
            this.getScope(scope).clear();
        }
    }
}

class ScopedStorage {
    constructor(scope, driver, ttl) {
        this.scope = scope;
        this.driver = driver;
        this.ttl = ttl;
        if (this.ttl === undefined) this.ttl = 60 * 60 * 1000; // 1 hour
    }

    getAll() {
        let storage = this;
        return this.keys().reduce(function (acc, item) {
            acc[item] = storage.getItem(item);
            return acc;
        }, {});
    }

    setItem(key, value) {
        let scopedKey = `${this.scope}-${key}`,
            scopedExpirationKey = `${this.scope}-${key}-expiration`;

        this.driver.setItem(scopedKey, JSON.stringify(value));
        this.driver.setItem(
            scopedExpirationKey,
            new Date().getTime() + this.ttl
        );
    }

    getItem(key) {
        let scopedKey = `${this.scope}-${key}`,
            scopedExpirationKey = `${this.scope}-${key}-expiration`,
            response = this.driver.getItem(scopedKey),
            expirationResponse = this.driver.getItem(scopedExpirationKey);

        if (response === undefined || response === 'undefined')
            return undefined;
        if (response === null || response === 'null') return null;

        response = JSON.parse(response);

        if (
            !isEmpty(expirationResponse) &&
            expirationResponse <= new Date().getTime()
        ) {
            this.removeItem(scopedKey);
            return null;
        }
        return response;
    }

    keys() {
        let scope = `${this.scope}-`,
            theseKeys;
        if (this.driver.keys !== undefined) {
            theseKeys = this.driver.keys();
        } else {
            theseKeys = Object.getOwnPropertyNames(this.driver);
        }
        return theseKeys
            .filter((key) => key.startsWith(scope))
            .map((key) => key.substring(scope.length))
            .filter((key) => key != 'expiration');
    }

    removeItem(key) {
        let scopedKey = `${this.scope}-${key}`;
        return this.driver.removeItem(scopedKey);
    }

    removePrefix(prefix) {
        for (let key of this.keys()) {
            if (key.startsWith(prefix)) {
                this.removeItem(key);
            }
        }
    }

    clear() {
        for (let key of this.keys()) {
            this.removeItem(key);
        }
        this.setItem('expiration', {});
        return this.driver.clear();
    }

    key(index) {
        return this.keys()[index];
    }
}

class CookieStorage {
    constructor() {
        this.expiration = new Date();
        this.expiration.setTime(
            this.expiration.getTime() + 30 * 24 * 60 * 60 * 1000
        );
    }

    keys() {
        let cookies = decodeURIComponent(document.cookie).split(';'),
            cookieName,
            cookieNames = [];

        for (let cookie of cookies) {
            cookieName = cookie.split('=')[0];
            while (cookieName.charAt(0) == ' ') {
                cookieName = cookieName.substring(1);
            }
            cookieNames.push(cookieName);
        }
        return cookieNames;
    }

    getItem(key) {
        let cookies = decodeURIComponent(document.cookie).split(';');
        for (let cookie of cookies) {
            while (cookie.charAt(0) == ' ') {
                cookie = cookie.substring(1);
            }
            if (cookie.startsWith(`${key}=`)) {
                return JSON.parse(cookie.substring(key.length + 1));
            }
        }
        return null;
    }

    setItem(key, value, expiration) {
        if (expiration === undefined) {
            expiration = this.expiration;
        }
        document.cookie = `${key}=${JSON.stringify(
            value
        )};expires=${expiration.toUTCString()};path=/;`;
    }

    removeItem(key) {
        let expiration = new Date();
        expiration.setTime(0);
        this.setItem(key, '', expiration);
    }

    clear() {
        for (let cookieName of this.keys()) {
            this.removeItem(cookieName);
        }
    }

    key(index) {
        return this.keys()[index];
    }
}

class MemoryStorage {
    constructor() {
        this.storage = {};
    }

    keys() {
        return Object.getOwnPropertyNames(this.storage);
    }

    setItem(key, value) {
        this.storage[key] = value;
    }

    getItem(key) {
        if (this.storage.hasOwnProperty(key)) {
            return this.storage[key];
        }
        return null;
    }

    removeItem(key) {
        delete this.storage[key];
    }

    removePrefix(prefix) {
        for (let key of this.keys()) {
            if (key.startsWith(prefix)) {
                this.removeItem(key);
            }
        }
    }

    clear() {
        for (let key of this.keys) {
            delete this.storage[key];
        }
    }

    key(index) {
        return this.keys()[index];
    }
}

let getDriver = function () {
    try {
        if (typeof Storage !== undefined) {
            return window.localStorage;
        }
    } catch (e) {
        console.error("Couldn't get local storage, defaulting to memory.", e);
        return new MemoryStorage();
    }
};

export { SessionStorage, MemoryStorage, CookieStorage, ScopedStorage };
export let Session = new SessionStorage(getDriver());
