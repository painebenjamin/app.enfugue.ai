/** @module model/index */
import { API, JSONAPI } from '../base/api.mjs';
import { getCookie, isEmpty, kebabCase, deepClone } from '../base/helpers.mjs';

/**
 * This parses datetimes to JS date objects opaquely.
 * @param string $value An ISO-8601 datetime.
 * @return mixed a Date object if the value passed is a date string, or the original value passed.
 */
let checkParseValue = (value) => {
    if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(.\d+)?$/.test(value)) {
        return new Date(value);
    }
    return value;
};

/**
 * Our API extends the base JSONAPI by allowing uploads (multipart)
 */
class ModelAPI extends JSONAPI {
    /**
     * Sends a multipart payload (allows files).
     *
     * @param string $url The url to send.
     * @param object $headers Header key-value pairs.
     * @param object $parameters Query string key-value pairs.
     * @param object $payload Key-value pairs, encoded multipart/form-encoding
     * @param callable $onProgess An optional callable to fire with progress events.
     * @param callable $includeMeta Whether or not to return metadata with your query,
     * @return Either [data, meta] or just data, depending on $includeMeta.
     */
    async multiPart(url, headers, parameters, payload, onProgress, includeMeta) {
        let formData = new FormData();
        for (let payloadPart in payload) {
            formData.append(payloadPart, payload[payloadPart]);
        }
        let timeout = this.timeout;
        this.timeout = null;
        let result = await API.prototype.query.apply(this, [
            'POST',
            url,
            headers,
            parameters,
            formData,
            onProgress,
            includeMeta
        ]);
        this.timeout = timeout;
        return result;
    }
}

/**
 * This allows for a small set of configuration to manufacture an ORM based
 * on the expected format of the Enfugue API.
 *
 * Extend this class with $apiRoot and $apiScope.
 */
class ModelObject {
    /**
     * @var bool Allow classes to designate that they should always ask for included references
     */
    static alwaysInclude = false;

    /**
     * The constructor is called by the ORM manager (the Model itself.)
     * We use underscored namespacing to try and avoid conflicts with user-defined properties.
     *
     * @param Model $model The model, which is constructing this.
     * @param object $attributes All attributes of the object.
     * @param object $relationships An optional set of relationships this object has. Use `apiInclude` to add these.
     * @param array $see Any number of other references issued by the API.
     */
    constructor(model, attributes, relationships, see) {
        this._model = model;
        this._attributes = {};
        for (let attributeName in attributes) {
            let attributeValue = attributes[attributeName];
            this._attributes[attributeName] = checkParseValue(attributeValue);
        }
        this._relationships = relationships;
        this._seeAlso = see;
        this._changes = {};
        this._dirty = false;
        this._deleted = false;

        for (let attributeName in attributes) {
            Object.defineProperty(this, attributeName, {
                get: () => {
                    return isEmpty(this._changes[attributeName])
                        ? this._attributes[attributeName]
                        : this._changes[attributeName];
                },
                set: (value) => {
                    if (this._deleted) {
                        throw 'Object is deleted, cannot modify.';
                    }
                    this._changes[attributeName] = value;
                    this._dirty = true;
                }
            });
        }

        if (!isEmpty(relationships)) {
            for (let relationshipName in relationships) {
                Object.defineProperty(this, relationshipName, {
                    get: () => {
                        return this._relationships[relationshipName];
                    }
                });
            }
        }
    }

    /**
     * @return array<object> Passes through the 'see also' array from the API.
     */
    get see() {
        return this._seeAlso;
    }

    /**
     * A simple passthrough to the attribute object
     *
     * @return object The attributes object.
     */
    getAttributes() {
        return this._attributes;
    }

    /**
     * Passes through a query to the model.
     *
     * @param string $method The method to use, usually GET but could be anything.
     * @param string $url The URL to download from, required.
     * @param object $headers Optional key-value pairs to send as headers.
     * @param object $payload Optional data to send along with the request.
     * @param callable $onProgress Optional function to be called with progress events.
     * @param bool $includeMeta Whether or not to include metadata in the response. Default false.
     * @return Either [data, meta] or just data, depending on $includeMeta.
     */
    queryModel(method, url, headers, parameters, payload, onProgress, includeMeta) {
        return this._model.query(method, url, headers, parameters, payload, onProgress, includeMeta);
    }

    /**
     * Given a set of filters, determine if it has all of the scope variables necessary
     * to describe a path to a single object for a standard REST structure.
     *
     * Ex: given `static apiScope = ["name"]`, this will return `true` if `filters` contains `name`.
     * 
     * @param object $filters The filters to look for.
     */
    static hasFullScope(filters) {
        let apiScope = this.apiScope;
        if (!isEmpty(this.apiScope)) {
            for (let scopePart of apiScope) {
                if (isEmpty(filters[scopePart])) return false;
            }
        }
        return true;
    }

    /**
     * @return array<string> An array of all callable properties of this object.
     */
    static getMethods() {
        return Object.getOwnPropertyNames(this).filter(
            (property) => typeof this[property] == 'function'
        );
    }

    /**
     * Given a set of filters, construct the API URL to this object or objects.
     *
     * @param object $filters An object of filters to try and put in the URL, if configured.
     * @return string The API URL.
     */
    static getURL(filters) {
        let apiRoot = this.apiRoot,
            apiScope = this.apiScope,
            apiInclude = this.apiInclude,
            apiFilters = isEmpty(filters) ? {} : deepClone(filters),
            hasFullScope = isEmpty(apiScope);

        if (isEmpty(apiRoot)) {
            apiRoot = kebabCase(this.name);
        }

        let requestUrl = `${apiRoot}/`,
            requestParameters = {};

        if (!isEmpty(apiScope) && !isEmpty(apiFilters)) {
            for (let scopePart of apiScope) {
                if (!isEmpty(apiFilters[scopePart])) {
                    requestUrl += `${apiFilters[scopePart]}/`;
                    delete apiFilters[scopePart];
                    hasFullScope = true;
                } else {
                    hasFullScope = false;
                    break;
                }
            }
        }

        if (!isEmpty(apiFilters)) {
            requestParameters['filter'] = Object.getOwnPropertyNames(apiFilters).map(
                (name) => `${name}:${apiFilters[name]}`
            );
        }
        
        if ((hasFullScope || this.alwaysInclude) && !isEmpty(apiInclude)) {
            requestParameters['include'] = apiInclude;
        }

        return [
            requestUrl.substring(0, requestUrl.length - 1),
            requestParameters
        ];
    }

    /**
     * @return string The URL to this individual object.
     */
    get url() {
        return this.constructor.getURL(this._attributes)[0];
    }

    /**
     * Puts an object full of changes on the stage to be patched later.
     * @param object $changes Key-value pairs to change on this object.
     */
    stageChanges(changes) {
        for (let changeKey in changes) {
            this.stageChange(changeKey, changes[changeKey]);
        }
    }

    /**
     * Stages a single change, preparing this to be PATCHed.
     * @param string $key The attribute key.
     * @param mixed $value The attribute value.
     */
    stageChange(key, value) {
        if (this._deleted) throw 'Object is deleted, cannot modify.';
        this._changes[key] = value;
        this._dirty = true;
    }

    /**
     * Saves any changes.
     * Throws an error if this object was deleted.
     * Transparently returns when no changes have been staged.
     */
    async save() {
        if (this._deleted) {
            throw 'Object is deleted.';
        }

        if (!this._dirty) {
            return;
        }

        let result = await this.queryModel("patch", this.url, {}, {}, this._changes);

        for (let changeName in this._changes) {
            this._attributes[changeName] = this._changes[changeName];
        }

        this._changes = {};
        this._dirty = false;
    }

    /**
     * Re-issue a query for a single object, with the expectation that it
     * has changed.
     *
     * @param object $additionalParameters Key-value pairs of any additional query string parameters.
     * @return ModelObject A new instance of this with the API parameters.
     */
    async requery(additionalParameters) {
        let [requestUrl, parameters] = this.constructor.getURL(this._attributes);
        delete parameters['filter'];
        if (!isEmpty(additionalParameters)) {
            parameters = { ...additionalParameters, ...parameters };
        }
        let result = await this._model.get(requestUrl, {}, parameters);
        return this.constructor.mapModelResult(this._model, result);
    }

    /**
     * Deletes an object in the API (sends a DELETE request.)
     */
    async delete() {
        if (this.deleted) {
            throw 'Object is deleted.';
        }
        let [requestUrl] = this.constructor.getURL(this._attributes);
        await this._model.delete(requestUrl);
        this._deleted = true;
    }

    /**
     * Creates a new model via POST. This is called by the Model.
     *
     * @param Model $model The model (caller).
     * @param object $attributes Key-value pairs for the new object.
     * @param callable $onProgress When using multipart, this will be called during upload.
     */
    static async create(model, attributes, onProgress) {
        let apiName = isEmpty(this.apiName) ? this.name : this.apiName,
            [requestUrl] = this.getURL(),
            method = this.multiPart === true ? model.multiPart : model.post,
            result = await method.call(
                model,
                requestUrl,
                {},
                {},
                attributes,
                onProgress
            );

        return this.mapModelResult(model, result);
    }

    /**
     * Finds one or many instances of a configured ModelObject.
     * Called by the Model.
     *
     * @param Model $model The model (caller)
     * @param object $filters The filters to pass in as parameters, optional.
     * @param object $additionalParameters Any number of additional key-value pairs to add to the request.
     * @param bool $includeMeta Whether or not to include metadata in the request.
     * @return Either [data, meta] or just data, depending on $includeMeta.
     */
    static async query(model, filters, additionalParameters, includeMeta) {
        let apiName = isEmpty(this.apiName) ? this.name : this.apiName,
            [requestUrl, requestParameters] = this.getURL(filters);
        requestParameters = {
            ...requestParameters,
            ...(isEmpty(additionalParameters) ? {} : additionalParameters)
        };
        let result = await model.get(requestUrl, {}, requestParameters, null, null, includeMeta),
            data,
            meta,
            mappedResult;
        
        if (includeMeta) {
            [data, meta] = result;
        } else {
            data = result;
        }

        mappedResult = this.mapModelResult(model, data);
        if (includeMeta) {
            return [mappedResult, meta];
        }
        return mappedResult;
    }

    /**
     * Given a raw API response, map to configured classes.
     * Called by the model.
     *
     * @param Model $model The model (caller)
     * @param object $result The result, either an array of results or one result.
     * @return ModelObject The instantiated model objects.
     */
    static mapModelResult(model, result) {
        let apiName = isEmpty(this.apiName) ? this.name : this.apiName,
            mapResultPart = (resultPart) => {
                let classConstructor;

                if (resultPart.type === apiName) {
                    classConstructor = this;
                } else {
                    classConstructor = model.constructor.modelObjects
                        .filter(
                            (object) =>
                                (isEmpty(object.apiName)
                                    ? object.name
                                    : object.apiName) === resultPart.type
                        )
                        .shift();
                }

                if (isEmpty(classConstructor)) {
                    throw 'Cannot map type ' + resultPart.type;
                }

                let resultAttributes = resultPart.attributes,
                    resultRelationships = resultPart.include,
                    resultSee = resultPart.see;

                if (!isEmpty(resultRelationships)) {
                    resultRelationships = Object.getOwnPropertyNames(
                        resultRelationships
                    ).reduce((carry, relationshipName) => {
                        let relationships =
                            resultRelationships[relationshipName];
                        if (
                            Array.isArray(relationships) &&
                            relationships.length > 0
                        ) {
                            carry[relationshipName] =
                                relationships.map(mapResultPart);
                        } else if (!isEmpty(relationships)) {
                            carry[relationshipName] =
                                mapResultPart(relationships);
                        }
                        return carry;
                    }, {});
                }
                if (!isEmpty(resultSee)) {
                    resultSee = resultSee.map((seePart) =>
                        mapResultPart(seePart)
                    );
                }
                return new classConstructor(
                    model,
                    resultAttributes,
                    resultRelationships,
                    resultSee
                );
            };

        if (isEmpty(result)) {
            return null;
        } else if (Array.isArray(result)) {
            if (result.length > 1) {
                return result.map(mapResultPart);
            }
            return mapResultPart(result[0]);
        } else {
            return mapResultPart(result);
        }
    }
}

/**
 * This simple class merges the ModelObject and Model to bind a ModelObject
 * to a Model. The Bound object is then what is accessed and used by the model users.
 */
class ModelBoundObject {
    /**
     * Construct a new ModelBoundObject.
     *
     * @param Model $model The model (caller).
     * @param class $modelObject The class extending ModelObject to bind.
     */
    constructor(model, modelObject) {
        this._model = model;
        this._modelObject = modelObject;
        for (let passThroughMethod of ['query', 'create'].concat(modelObject.getMethods())) {
            this[passThroughMethod] = function () {
                return modelObject[passThroughMethod].apply(
                    modelObject,
                    [model].concat(Array.from(arguments))
                );
            };
        }
    }

    /**
     * A small extension of the Model creates a way to be sure that no limits are passed.
     *
     * @param object $filters An optional array of filters.
     * @return array<ModelBoundObject> An array of results.
     */
    async queryAll(filters) {
        let result = await this.query(filters, 0);
        if (isEmpty(result)) {
            result = [];
        } else if (!Array.isArray(result)) {
            result = [result];
        }
        return result;
    }
}

/**
 * The Model class instantiates the API, registers model objects, and ensures that
 * API keys and/or authentication tokens are passed to all API calls.
 */
class Model {
    /**
     * @var array The objects to bind to this model.
     */
    static modelObjects = [];

    /**
     * @var array<string> Methods that can be debounced.
     */
    static debounceMethods = ['get', 'head', 'options'];

    /**
     * @param object $configuation The root configuration object, required for URLs.
     */
    constructor(configuration) {
        let url = configuration.url.api;
        if(!url.startsWith("http")){
            url = configuration.url.base;
            if(url.endswith("/")){
                if(configuration.url.api.startswith("/")){
                    url = url.substring(0, url.length-1);
                }
            } else if (!configuration.url.api.startswith("/")) {
                url += "/";
            }
            url += configuration.url.api;
        }
        this.api = new ModelAPI(
            url,
            configuration.debug
        );
        this.key = isEmpty(configuration.keys) ? null : configuration.keys.api;
        this.token = getCookie(configuration.model.cookie.name);
        this.debounce = {};
        for (let methodName of API.allMethods) {
            this[methodName] = (url, headers, parameters, payload, onProgress, includeMeta) =>
                this.query(
                    methodName.toUpperCase(),
                    url,
                    headers,
                    parameters,
                    payload,
                    onProgress,
                    includeMeta
                );
            if (this.constructor.debounceMethods.indexOf(methodName) !== -1) {
                this.debounce[methodName] = {};
            }
        }
        for (let modelObject of this.constructor.modelObjects) {
            this[modelObject.name] = new ModelBoundObject(this, modelObject);
        }
    }

    /**
     * Downloads a file.
     *
     * @param string $method The method to use, usually GET but could be anything.
     * @param string $url The URL to download from, required.
     * @param object $headers Optional key-value pairs to send as headers.
     * @param object $payload Optional data to send along with the request.
     * @param callable $onProgress Optional function to be called with progress events.
     * @return mixed The response.
     */
    async download(method, url, headers, parameters, payload, onProgress) {
        headers = headers || {};
        headers["Authorization"] = "Bearer " + this.token;
        if (!isEmpty(this.key)) {
            headers["X-API-Key"] = this.key;
        }
        let result = await this.api.download(
            method,
            url,
            headers,
            parameters,
            payload,
            onProgress
        );
        return result;
    }

    /**
     * Use the API to query for object(s).
     *
     * @param string $method The method to use, usually GET but could be anything.
     * @param string $url The URL to download from, required.
     * @param object $headers Optional key-value pairs to send as headers.
     * @param object $payload Optional data to send along with the request.
     * @param callable $onProgress Optional function to be called with progress events.
     * @param bool $includeMeta Whether or not to include metadata in the response. Default false.
     * @return Either [data, meta] or just data, depending on $includeMeta.
     */
    query(method, url, headers, parameters, payload, onProgress, includeMeta) {
        method = method.toLowerCase();
        let isDebounceable = this.constructor.debounceMethods.indexOf(method) !== -1,
            fullUrl;

        if (isDebounceable) {
            fullUrl = this.api.buildUrl(url, parameters);
            if (this.debounce[method][fullUrl] !== undefined) {
                console.warn(`Debouncing ${method.toUpperCase()} ${fullUrl}`);
                return this.debounce[method][fullUrl];
            }
        }

        headers = headers || {};
        headers["Authorization"] = "Bearer " + this.token;
        if (!isEmpty(this.key)) {
            headers["X-API-Key"] = this.key;
        }

        let promise = new Promise(async (resolve, reject) => {
            try {
                let result = await this.api.query(
                    method,
                    url,
                    headers,
                    parameters,
                    payload,
                    onProgress
                );
                if (includeMeta) {
                    resolve([result.data, result.meta]);
                } else {
                    resolve(result.data);
                }
            } catch (e) {
                try {
                    reject(e.errors[0]);
                } catch (e2) {
                    console.error(
                        "Error in API request",
                        method,
                        url,
                        headers,
                        parameters,
                        payload
                    );
                    reject(e);
                }
            } finally {
                if (isDebounceable) {
                    delete this.debounce[method][fullUrl];
                }
            }
        });

        if (isDebounceable) {
            this.debounce[method][fullUrl] = promise;
        }
        return promise;
    }
    
    /**
     * Use the API to query for anything other than object(s), skipping mapping.
     *
     * @param string $method The method to use, usually GET but could be anything.
     * @param string $url The URL to download from, required.
     * @param object $headers Optional key-value pairs to send as headers.
     * @param object $payload Optional data to send along with the request.
     * @param callable $onProgress Optional function to be called with progress events.
     * @return mixed The response.
     */
    rawQuery(method, url, headers, parameters, payload, onProgress) {
        headers = headers || {};
        headers["Authorization"] = "Bearer " + this.token;
        if (!isEmpty(this.key)) {
            headers["X-API-Key"] = this.key;
        }
        return this.api.rawQuery(method, url, headers, parameters, payload, onProgress);
    }

    /**
     * Sends a POST request in multipart format instead of JSON.
     *
     * @param string $url The URL to download from, required.
     * @param object $headers Optional key-value pairs to send as headers.
     * @param object $payload Optional data to send along with the request.
     * @param callable $onProgress Optional function to be called with progress events.
     * @return mixed The response.
     */
    async multiPart(url, headers, parameters, payload, onProgress) {
        headers = headers || {};
        headers["Authorization"] = "Bearer " + this.token;
        if (!isEmpty(this.key)) {
            headers["X-API-Key"] = this.key;
        }
        try {
            let result = await this.api.multiPart(
                url,
                headers,
                parameters,
                payload,
                onProgress
            );
            return JSON.parse(result).data;
        } catch (e) {
            try {
                throw JSON.parse(e.responseText).errors[0].detail;
            } catch (e2) {
                throw e;
            }
        }
    }
}

export { Model, ModelObject };
