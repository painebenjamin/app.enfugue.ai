/** @module base/api */
import { isEmpty } from "./helpers.mjs";

/** Represents a persistent state container for API interactions. */
class API {
    /**
     * @var Array<string> All request methods
     */
    static allMethods = [
        'get',
        'put',
        'post',
        'delete',
        'patch',
        'head',
        'options',
        'trace',
        'connect'
    ];

    /**
     * @var int The global request timeout (when not uploading)
     */
    static requestTimeout = 90000;

    /**
     * Constructs an API container.
     * @param {string} The base URL. Optional.
     * @param {bool}   true = log requests/responses to console.
     */
    constructor(baseUrl, debug) {
        this.timeout = this.constructor.requestTimeout;
        if (!baseUrl) {
            this.baseUrl = '';
        } else {
            this.baseUrl = encodeURI(
                baseUrl.endsWith('/')
                    ? baseUrl.substring(0, baseUrl.length - 1)
                    : baseUrl
            );
        }
        this.debug = debug === true;
        for (let methodName of this.constructor.allMethods) {
            /* For each default method, assign a quick partial for easy method
             * calls on each verb (e.g. post, get, etc.) */
            this[methodName] = (
                url,
                headers,
                parameters,
                payload,
                onProgress
            ) =>
                this.query(
                    methodName,
                    url,
                    headers,
                    parameters,
                    payload,
                    onProgress
                );
        }
    }

    /**
     * Given a URL and optional parameters, construct a full URL.
     * @param {string} Either a relative URL to concatenate to this.baseUrl, or absolutely URL.
     * @param {object} Any query parameters to encode to the end of the URL.
     * @return {string} The full URL.
     */
    buildUrl(url, parameters) {
        let requestUrl;
        if (url === undefined) url = '';
        if (url === undefined) {
            requestUrl = this.baseUrl;
        } else if (Array.isArray(url)) {
            requestUrl = [this.baseUrl]
                .concat(url.map((urlPart) => encodeURIComponent(urlPart)))
                .join('/');
        } else {
            if (url.startsWith('http')) {
                requestUrl = url;
            } else if (url.startsWith('//')) {
                let secured = window.location.href.substring(0, 5) === 'https';
                requestUrl = `${secured ? 'https' : 'http'}://${url}`;
            } else {
                if (url.startsWith('/')) {
                    url = url.substring(1);
                }
                requestUrl = `${this.baseUrl}/${encodeURI(url)}`;
            }
        }
        if (
            parameters !== undefined &&
            Object.getOwnPropertyNames(parameters).length > 0
        ) {
            let parameterString = Object.getOwnPropertyNames(parameters)
                .map((parameter) =>
                    Array.isArray(parameters[parameter])
                        ? parameters[parameter]
                              .map(
                                  (value) =>
                                      `${parameter}=${encodeURIComponent(
                                          value
                                      )}`
                              )
                              .join('&')
                        : `${parameter}=${encodeURIComponent(
                              parameters[parameter]
                          )}`
                )
                .join('&');
            requestUrl = `${requestUrl}?${parameterString}`;
        }
        return requestUrl;
    }

    /**
     * The meat of the API; this builds a query, executes it, and returns the result.
     * This uses the Promise() interface, but most implementing usage should use await.
     *
     * @param   {string}    The HTTP verb
     * @param   {string}    The URL, either an absolute URL or a relative one
     * @param   {object}    Parameters to build a query string with
     * @param   {object}    A payload to send.
     * @param   {func}      Optionally, a method to call on upload progress events.
     * @return  {Promise}   A promise object that can be using in async functions.
     */
    query(method, url, headers, parameters, payload, onProgress) {
        headers = headers || {};
        parameters = parameters || {};
        payload = payload || {};
        onProgress = onProgress || (() => {});
        method = method.toUpperCase();

        let debug = this.debug,
            timeout = this.timeout;

        return new Promise((resolve, reject) => {
            let request = new XMLHttpRequest(),
                requestUrl = this.buildUrl(url, parameters);

            request.addEventListener('load', function () {
                if (debug) {
                    console.log(
                        'Response from',
                        requestUrl,
                        ':',
                        this.responseText
                    );
                }
                if (
                    this.readyState === 4 &&
                    this.status >= 200 &&
                    this.status < 400
                ) {
                    resolve(this.responseText);
                } else {
                    reject(this);
                }
            });

            request.addEventListener('error', function (e) {
                reject(this);
            });

            request.addEventListener('timeout', function (e) {
                reject(this);
            });

            request.upload.onprogress = onProgress;
            request.timeout = timeout;

            if (debug) {
                console.log(method, 'Request to', requestUrl, payload);
            }

            try {
                request.open(method, requestUrl);
                for (let headerName in headers) {
                    request.setRequestHeader(headerName, headers[headerName]);
                }
                request.send(isEmpty(payload) ? undefined : payload);
            } catch (e) {
                reject(request, e);
            }
        });
    }

    download(method, url, headers, parameters, payload) {
        headers = headers || {};
        parameters = parameters || {};
        payload = payload || {};
        return new Promise((resolve, reject) => {
            let request = new XMLHttpRequest(),
                requestUrl = this.buildUrl(url, parameters);

            request.responseType = 'blob';
            request.addEventListener('load', function () {
                if (
                    this.readyState === 4 &&
                    this.status >= 200 &&
                    this.status < 400
                ) {
                    let blob = this.response,
                        contentDisposition = this.getResponseHeader(
                            'Content-Disposition'
                        ),
                        fileName = contentDisposition.match(
                            /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/
                        )[1],
                        anchor = document.createElement('a');
                    if (fileName.startsWith('"') || fileName.startsWith("'")) {
                        fileName = fileName.substring(1);
                    }
                    if (fileName.endsWith('"') || fileName.endsWith("'")) {
                        fileName = fileName.substring(0, fileName.length - 1);
                    }
                    anchor.href = window.URL.createObjectURL(blob);
                    anchor.download = fileName;
                    anchor.dispatchEvent(new MouseEvent('click'));
                    resolve();
                } else {
                    reject(this);
                }
            });
            try {
                request.open(method, requestUrl);
                for (let headerName in headers) {
                    request.setRequestHeader(headerName, headers[headerName]);
                }
                request.send(payload);
            } catch (e) {
                reject(request, e);
            }
        });
    }
}

/**
 * The JSONAPI extends the regular API but adds some methods for JSON parsing.
 */
class JSONAPI extends API {
    /**
     * If there is a payload, set headers and encode to JSON.
     */
    async download(method, url, headers, parameters, payload) {
        headers = headers || {};

        if (['POST', 'PUT', 'PATCH'].indexOf(method.toUpperCase()) != -1) {
            headers['Content-Type'] = 'application/json';
            payload =
                payload === undefined || payload === null
                    ? null
                    : JSON.stringify(payload);
        }

        return super.download(method, url, headers, parameters, payload);
    }

    /**
     * Set rawQuery to redirect to parent's base query.
     */
    rawQuery(method, url, headers, parameters, payload, onProgress) {
        return super.query(
            method,
            url,
            headers,
            parameters,
            payload,
            onProgress
        );
    }

    /**
     * Encode/decode when querying with JSONAPI.
     */
    async query(method, url, headers, parameters, payload, onProgress) {
        headers = headers || {};

        if (['POST', 'PUT', 'PATCH'].indexOf(method.toUpperCase()) != -1 && !isEmpty(payload)) {
            headers['Content-Type'] = 'application/json';
        }
        
        let response,
            parsedResponse,
            isError = false;

        try {
            response = await super.query(
                method,
                url,
                headers,
                parameters,
                isEmpty(payload) ? null : JSON.stringify(payload),
                onProgress
            );
        } catch (e) {
            isError = true;
            response = e.responseText;
        }

        try {
            parsedResponse = JSON.parse(response);
        } catch (e) {
            parsedResponse = response;
            if (this.debug) {
                console.warn('Couldn\'t parse "' + response + '"');
                console.error(e);
            }
        }
        if (isError) {
            throw parsedResponse;
        }
        return parsedResponse;
    }
}

export { API, JSONAPI };
