let eventFired = false,
    loadedCallbacks = [];

let fireEvents = function () {
    for (let callback of loadedCallbacks) {
        callback();
    }
};

let documentReady = new Promise(function (resolve, reject) {
    if (
        document.readyState &&
        (document.readyState === 'complete' ||
            document.readyState === 'loaded' ||
            document.readyState === 'interactive')
    ) {
        resolve();
    } else if (document.addEventListener) {
        let onDOMContentLoaded = function () {
            resolve();
            document.removeEventListener(
                'DOMContentLoaded',
                onDOMContentLoaded,
                false
            );
        };
        document.addEventListener('DOMContentLoaded', onDOMContentLoaded);
    } else if (document.attachEvent) {
        let onReadyStateChange = function () {
            if (document.readyState === 'complete') {
                resolve();
                document.detachEvent('onreadystatechange', onReadyStateChange);
            }
        };
        document.attachEvent('onreadystatechange', onReadyStateChange);
        if (document.documentElement.doScroll && window == window.top) {
            var tryScrolling = function () {
                try {
                    document.documentElement.doScroll('left');
                    resolve();
                } catch (e) {
                    setTimeout(tryScrolling, 0);
                    return;
                }
            };
            tryScrolling();
        }
    }
});

documentReady.then(function () {
    eventFired = true;
    fireEvents();
});

export let Loader = {
    done: function (callback) {
        if (eventFired) {
            callback();
        } else {
            loadedCallbacks.push(callback);
        }
    }
};
