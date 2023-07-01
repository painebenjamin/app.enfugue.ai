/** @module base/helpers */

/**
 * Asserts all in an array is truthy
 *
 * @param array<mixed> An array of values to assert are truthy
 * @return true if all values are truthy
 */
export let every = (values) => {
    for (let value of values) {
        if (!value) return false;
    }
    return true;
};

/**
 * Asserts all in an array is falsey
 *
 * @param array<mixed> An array of values to assert are falsey
 * @return true if all values are falsey
 */
export let none = (values) => {
    for (let value of values) {
        if (value) return false;
    }
    return true;
};

/**
 * Asserts any in an array are truthy
 *
 * @param array<mixed> An array of values to assert are truthy
 * @return true if any values are truthy
 */
export let any = (values) => {
    for (let value of values) {
        if (value) return true;
    }
    return false;
};

/**
 * Flattens a multidimensionsal array into a monodimensional one
 *
 * @param array $arr The array to flatten
 * @return array A monodimensional array
 */
export let flatten = (arr) => {
    return arr.reduce(
        (carry, item) =>
            carry.concat(Array.isArray(item) ? flatten(item) : item),
        []
    );
};

/**
 * Removes leading and trailing white space or quotes from a string
 *
 * @param string $str The string the strip
 * @return string The stripped string
 */
export let strip = (str) => {
    if (isEmpty(str)) return str;
    let match = str.match(/^[\ \t\r\n\'\"]*(.*?)[\ \t\r\n\'\"]*$/);
    if (match === null) return "";
    return match[1];
};

/**
 * Returns a promise that is resolved in a specified time
 *
 * @param int $ms The number of milliseconds to sleep for.
 * @return Promise a promise that is resolved when the sleep ends.
 */
export let sleep = (ms) =>
    new Promise((resolve) => {
        setTimeout(resolve, ms);
    });

/**
 * Waits for a specific condition to occur.
 * Uses a polling interval.
 *
 * @param callable $condition The condition to check.
 * @param object $options An object containing interval and timeout.
 * @return Promise A promise that is resolvedwhen the condition is met.
 */
export let waitFor = async (condition, options) => {
    options = options || {};
    let interval = options.interval || 50,
        timeout = options.timeout,
        start = new Date(), 
        waited = 0;

    while (!condition()) {
        await sleep(interval);
        waited = new Date() - start;
        if (!isEmpty(timeout) && waited > timeout) {
            throw "Timeout";
        }
    }
};

/**
 * Given a string, guess the case.
 *
 * @param string $str The string to check.
 * @return string The guessed case
 */
export let guessCase = (str) => {
    str = strip(str);
    return /^$/.test(str)
        ? "EMPTY"
        : /^[a-z09]+$/.test(str)
        ? "LOWER"
        : /^[A-Z09]+$/.test(str)
        ? "UPPER"
        : /^[A-Za-z0-9]+(_[A-Za-z0-9]+)+$/.test(str)
        ? "SNAKE"
        : /^[A-Za-z0-9]+(-[A-Za-z0-9]+)+$/.test(str)
        ? "KEBAB"
        : /^[a-z]+([A-Z][a-z0-9]+)+$/.test(str)
        ? "CAMEL"
        : /^([A-Z][a-z0-9]+)+$/.test(str)
        ? "PASCAL"
        : /^(\w+)((\W\w+)+)$/.test(str)
        ? "SENTENCE"
        : "UNKNOWN";
};

/**
 * Given a string, split it into its constituent parts.
 * Guesses the case first. Some examples:
 *      `guessStringParts("my-string")  == ["my", "string"]`
 *      `guessStringParts("myString")   == ["my", "string"]`
 *      `guessStringParts("my_string")  == ["my", "string"]`
 *      `guessStringParts("MyString")   == ["my", "string"]`
 *      `guessStringParts("My String")  == ["my", "string"]`
 *
 * @param string $str The string to split
 * @return array<string> The parts of the string.
 */
export let guessStringParts = (str) => {
    str = strip(str);
    let strCase = guessCase(str);
    switch (strCase) {
        case "SENTENCE":
            return str.split(" ").map((s) => s.toLowerCase());
        case "SNAKE":
            return str.split("_").map((s) => s.toLowerCase());
        case "KEBAB":
            return str.split("-").map((s) => s.toLowerCase());
        case "CAMEL":
            return str
                .split(/([A-Z])/g)
                .reduce(
                    (carry, item, i) =>
                        i == 0 || i % 2 == 1
                            ? carry.concat([item])
                            : carry.concat([carry.splice(-1)[0] + item]),
                    []
                )
                .map((s) => s.toLowerCase());
        case "PASCAL":
            return str
                .split(/([A-Z])/g)
                .reduce(
                    (carry, item, i) =>
                        i == 0
                            ? carry
                            : i % 2 == 1
                            ? carry.concat([item])
                            : carry.concat([carry.splice(-1)[0] + item]),
                    []
                )
                .map((s) => s.toLowerCase());
        case "EMPTY":
        case "LOWER":
        case "UPPER":
        case "UNKNOWN":
        default:
            return [str.toLowerCase()];
    }
};

/**
 * Turns a string to `kebab-case`
 *
 * @param string $str The string to format
 * @param string $separator The separator. Defaults to "-"
 * @return string The formatted string
 */
export let kebabCase = (str, separator) => {
    if (separator === undefined) separator = "-";
    return guessStringParts(str).join(separator);
};

/**
 * Turns a string to `snake_case`
 *
 * @param string $str The string to format
 * @param string $separator The separator. Defaults to "_"
 * @return string The formatted string
 */
export let snakeCase = (str, separator) => {
    if (separator === undefined) separator = "_";
    return guessStringParts(str).join(separator);
};

/**
 * Turns a string to `camelCase`
 *
 * @param string $str The string to format
 * @param string $separator The separator. Defaults to ""
 * @return string The formatted string
 */
export let camelCase = (str, separator) => {
    if (separator === undefined) separator = "";
    return guessStringParts(str)
        .map((s, i) => (i == 0 ? s : s[0].toUpperCase() + s.substr(1)))
        .join(separator);
};

/**
 * Turns a string to `PascalCase`
 *
 * @param string $str The string to format
 * @param string $separator The separator. Defaults to ""
 * @return string The formatted string
 */
export let pascalCase = (str, separator) => {
    if (separator === undefined) separator = "";
    return guessStringParts(str)
        .map((s) => s[0].toUpperCase() + s.substr(1))
        .join(separator);
};

/**
 * Turns a string to `Title Case`
 *
 * @param string $str The string to format.
 * @return string The formatted string
 */
export let titleCase = (str) => pascalCase(str, " ");

/**
 * Checks if two strings are equal in a case-insensitive fashion.
 * Examples of matches:
 *      `caseInsensitiveMatch("HELLO WORLD", "helloWorld")`
 *      `caseInsensitiveMatch("hello_world", "HELLO-WORLD")`
 */
export let caseInsensitiveMatch = (str1, str2) => {
    return (
        str1.toLowerCase() === str2.toLowerCase() ||
        kebabCase(str1) === kebabCase(str2)
    );
};

/**
 * Truncates a string.
 *
 * @param string $str The string to truncate
 * @param int $length The length of the string to truncate to
 * @param string $postfix What to append to the string when truncating, default ellipses.
 * @return string The truncated string.
 */
export let truncate = (str, length, postfix = "…") => {
    if (str.length <= length) {
        return str;
    }
    return str.substring(0, str.lastIndexOf(" ", length)) + postfix;
};

/**
 * Filters an array to remove duplicates.
 *
 * @param array An array of items.
 * @return array An array of items without duplicates
 */
export let set = (arr) => {
    return arr.filter((v, i) => arr.indexOf(v) === i);
};

/**
 * Merges any number of objects together recursively.
 *
 * @param object Any number of objects to merge together.
 * @return object The merged object.
 */
export let merge = function () {
    let allObjects = Array.from(arguments),
        merged = { ...allObjects[0] };
    for (let i = 1; i < allObjects.length; i++) {
        let otherObject = allObjects[i];
        for (let key in otherObject) {
            if (merged[key] === undefined) {
                merged[key] = otherObject[key];
            } else if (
                typeof merged[key] == 'object' &&
                typeof otherObject[key] == 'object'
            ) {
                merged[key] = merge(merged[key], otherObject[key]);
            } else {
                merged[key] = otherObject[key];
            }
        }
    }
    return merged;
};

/**
 * @var array<string> Events that are considered 'MouseEvents'
 */
const mouseEvents = [
    "click",
    "dblclick",
    "mouseenter",
    "mouseleave",
    "mouseup",
    "mousedown",
    "mousemove",
    "mouseover",
    "mouseout",
    "contextmenu"
];

/**
 * Creates an Event or MouseEvent.
 *
 * @param string $name The name of the event.
 * @return Event The appropriate kind of event.
 */
export let createEvent = (name) => {
    return typeof MouseEvent === "function" && mouseEvents.indexOf(name) > -1
        ? new MouseEvent(name)
        : typeof Event === "function"
        ? new Event(name)
        : (() => {
              let e = document.createEvent("Event");
              e.initEvent(name, true, true);
              return e;
          })();
};

/**
 * Gets a random integer.
 * Allows either single value syntax for an upper bound, or two for a lower and upper.
 *
 * @param int $l The lower bound or upper bound
 * @param int $u The upper bound.
 * @return int The random integer
 */
export let randomInt = (l, u) => ((u === undefined ? 0 : l) + Math.random() * (u === undefined ? l : u - l)) | 0;

/**
 * Gets any number of random hexadecimal values.
 *
 * @param n The number of values to generate.
 * @return string A string of the hex values joined together.
 */
export let randomHex = (n) => isEmpty(n)
    ? randomInt(16).toString(16)
    : new Array(n).fill(null).map(randomHex).join("");

/**
 * Gets a random choice from an array.
 *
 * @param array $arr The array to choose from.
 * @return mixed A random choice from that array.
 */
export let randomChoice = (arr) => arr[randomInt(arr.length)];

/**
 * Gets a UUID4-formatted UUID.
 *
 * @return string The UUID.
 */
export let uuid = () => `${randomHex(8)}-${randomHex(4)}-4${randomHex(3)}-${randomChoice(['8', '9', 'a', 'b'])}${randomHex(3)}-${randomHex(12)}`;

/**
 * Check if two values are equivalent.
 *
 * Differs from normal in that it checks for value comparison of key-value maps.
 * @param mixed $a The LHS
 * @param mixed $b The RHS
 * @return bool True if the values are equivalent.
 */
export let isEquivalent = (a, b) => {
    if (typeof a === typeof b) {
        if (Array.isArray(a) && Array.isArray(b)) {
            if (a.length !== b.length) return false;
            for (let i = 0; i < a.length; i++) {
                if (!isEquivalent(a[i], b[i])) return false;
            }
            return true;
        } else if (typeof a === "object") {
            if (a === null) {
                return b === null;
            }
            if (b === null) return false;
            let aKeys = Object.getOwnPropertyNames(a),
                bKeys = Object.getOwnPropertyNames(b);

            if (!isEquivalent(aKeys, bKeys)) return false;
            for (let key of aKeys) {
                if (!isEquivalent(a[key], b[key])) {
                    return false;
                }
            }
            return true;
        } else {
            return a === b;
        }
    }
    return false;
};

/**
 * Gets the query parameters from the current URL.
 *
 * @return object Current query parameters.
 */
export let getQueryParameters = () => {
    return window.location.href.indexOf("?") === -1
        ? {}
        : window.location.href
              .substring(window.location.href.indexOf("?") + 1)
              .split('&')
              .reduce((agg, item) => {
                  let itemParts = item.split('='),
                      itemKey = itemParts[0],
                      itemValue = decodeURIComponent(itemParts[1]).replace(
                          '+',
                          ' '
                      );
                  if (agg.hasOwnProperty(itemKey)) {
                      if (!Array.isArray(agg[itemKey])) {
                          agg[itemKey] = [agg[itemKey]];
                      }
                      agg[itemKey].push(itemValue);
                  } else {
                      agg[itemKey] = itemValue;
                  }
                  return agg;
              }, {});
};

/**
 * Gets all `data-` attributes from a node.
 *
 * @param DocumentElement $node The node to inspect.
 * @return object A key-value map of all data attributes turned to camelCase.
 */
export let getDataParameters = (node) => {
    let parameters = {};
    for (let i = 0, atts = node.attributes, n = atts.length; i < n; i++) {
        if (atts[i].nodeName.startsWith('data')) {
            parameters[camelCase(atts[i].nodeName.substr(5))] = atts[i].nodeValue;
        }
    }
    return parameters;
};

/**
 * Given a URL and parameters, build a full URL.
 *
 * @param string $url The base URL
 * @param object $parameters The parameters to encode in the URL.
 */
export let buildQueryURL = (url, parameters) => {
    let fullUrl = url;

    if (url.indexOf('?') === -1) {
        fullUrl += '?';
    }
    return (
        fullUrl +
        Object.getOwnPropertyNames(parameters)
            .map((item) => `${item}=${encodeURIComponent(parameters[item])}`)
            .join('&')
    );
};

/**
 * Gets the current URL with a different set of query parameters.
 *
 * @param object $parameters The parameters to put in the URL.
 * @return string The URL with query parameters
 */
export let getQueryURL = (parameters) => {
    return buildQueryURL(window.location.href, parameters);
};

/**
 * Check if an object is 'empty'.
 *
 * @param object $o The object to check.o
 * @return bool True if the object is empty.
 */
export let isEmpty = (o) => {
    return (
        o === null ||
        o === undefined ||
        o === '' ||
        o === 'null' ||
        (Array.isArray(o) && o.length === 0) ||
        (typeof o === 'object' &&
            o.constructor.name === 'Object' &&
            Object.getOwnPropertyNames(o).length === 0)
    );
};

/**
 * Removes empty array items and properties.
 * 
 * @param mixed $o The object to remove empties from.
 * @return mixed The object with empty properties or items removed.
 */
export let removeEmpty = (o) => {
    if (Array.isArray(o)) {
        let a = [];
        for (let value of o) {
            if (!isEmpty(v)) {
                a.push(v);
            }
        }
        return a;
    } else if (typeof o === 'object') {
        let a = {};
        for (let k of Object.getOwnPropertyNames(o)) {
            if (!isEmpty(o[k])) {
                a[k] = o[k];
            }
        }
        return a;
    }
    return o;
};

/**
 * Create a copy of whatever is passed.
 *
 * @param mixed $o An object to copy.
 * @return mixed The copied object.
 */
export let deepClone = (o, throwErrorOnFunction = true) => {
    if (o === undefined || o === null) return o;
    else if (typeof o === 'function'){
        if(throwErrorOnFunction) throw 'Cannot clone functions.';
        return o;
    }
    else if (typeof o === 'boolean') return !!o;
    else if (typeof o === 'number') return 0 + o;
    else if (typeof o === 'string' || o instanceof String) return `${o}`;
    else if (Array.isArray(o))
        return new Array(o.length).fill(null).map((_, i) => deepClone(o[i], false));
    else if (typeof o === 'object' || o === Object(o)) {
        switch (o.constructor.name) {
            case 'RegExp':
                return o;
            default:
                return Object.getOwnPropertyNames(o).reduce((carry, item) => {
                    carry[item] = deepClone(o[item], false);
                    return carry;
                }, new Object());
        }
    }
    console.warn('Unknown clone type for', o);
    return o;
};

/**
 * @var array<string> Suffixes for numbers of bytes in 1,000s
 */
const byteSuffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

/**
 * Formats a number of bytes as a better size - KB, MB, etc.
 *
 * @param int $bytes The number of bytes to format.
 * @param int $precision Optional, the float precision. Default 2.
 */
export let humanSize = (bytes, precision) => {
    if (precision === undefined) precision = 2;
    let suffixIndex = 0;
    while (bytes > 1000) {
        bytes /= 1000;
        suffixIndex++;
    }
    return `${bytes.toFixed(precision)} ${byteSuffixes[suffixIndex]}`;
};

const secondsPerMinute = 60,
      secondsPerHour = secondsPerMinute * 60,
      secondsPerDay = secondsPerHour * 24;

/**
 * Formats a number of seconds to a duration string.
 *
 * @param int $seconds The number of seconds to format.
 * @param bool $trim Whether or not to trim to first nonzero value, default true
 * @return string The formatted duration string
 */
export let humanDuration = (seconds, trim = true) => {
    let days = 0, hours = 0, minutes = 0;
    if (seconds > secondsPerDay) {
        days = Math.floor(seconds / secondsPerDay);
        seconds -= days * secondsPerDay;
    }
    if (seconds > secondsPerHour) {
        hours = Math.floor(seconds / secondsPerHour);
        seconds -= hours * secondsPerHour;
    }
    if (seconds > secondsPerMinute) {
        minutes = Math.floor(seconds / secondsPerMinute);
        seconds -= minutes * secondsPerMinute;
    }
    seconds = Math.floor(seconds);
    let durationString = `${days} day${days!=1?'s':''}, ${hours} hour${hours!=1?'s':''}, ${minutes} minute${minutes!=1?'s':''}, ${seconds} second${seconds!=1?'s':''}`;
    if (trim && days == 0) {
        let durationParts = durationString.split(", ");
        if (hours == 0 && minutes == 0) {
            return durationParts.slice(3).join(", ");
        } else if (hours == 0) {
            return durationParts.slice(2).join(", ");
        } else {
            return durationParts.slice(1).join(", ");
        }
    }
    return durationString;
};

/**
 * Gets the documents cookies as an object.
 *
 * @return object Document cookies
 */
export let getCookies = () => {
    return document.cookie.split(';').reduce((carry, item) => {
        let itemParts = strip(item).split('=');
        carry[itemParts[0]] = itemParts[1];
        return carry;
    }, {});
};

/**
 * Gets a single cookie
 *
 * @param string $cookieName The name of the cookie to get
 */
export let getCookie = (cookieName) => {
    return getCookies()[cookieName];
};

/**
 * The Jaro-Winkler likeness algorithm
 *
 * @param string $a The first string to check
 * @param string $b The second string to check
 * @param float $p The winkler immediacy multipler
 * @return float a Likeness between 0 and 1
 */
export let jaroWinkler = (a, b, p) => {
    let m = 0, i, j;
    if (a.length === 0 || b.length === 0) return 0;
    if (a === b) return 1;

    let range = Math.floor(Math.max(a.length, b.length) / 2) - 1,
        aMatches = new Array(a.length),
        bMatches = new Array(b.length);

    for (i = 0; i < a.length; i++) {
        let low = i >= range ? i - range : 0,
            high = i + range <= b.length - 1 ? i + range : b.length - 1;
        for (j = low; j <= high; j++) {
            if (aMatches[i] !== true && bMatches[j] !== true && a[i] === b[j]) {
                ++m;
                aMatches[i] = bMatches[j] = true;
                break;
            }
        }
    }

    if (m === 0) return 0; // No matches.

    let k = 0, transpositions = 0;

    for (i = 0; i < a.length; i++) {
        if (aMatches[i] === true) {
            for (j = k; j < b.length; j++) {
                if (bMatches[j] === true) {
                    k = j + 1;
                    break;
                }
            }
            if (a[i] !== b[j]) {
                ++transpositions;
            }
        }
    }

    let weight =
            (m / a.length + m / b.length + (m - transpositions / 2) / m) / 3,
        l = 0;

    if (p === undefined || p === null) p = 0.1;

    if (weight > 0.7) {
        while (a[l] === b[l] && l < 4) {
            ++l;
        }
        weight += l * p * (1 - weight);
    }

    return weight;
};

/**
 * Given an array, get all permutations.
 *
 * @param array $permutation The array to permuted
 * @return array An array of all permutations of the original array
 */
export let getPermutations = (permutation) => {
    let length = permutation.length,
        result = [permutation.slice()],
        c = new Array(length).fill(0),
        i = 1,
        k,
        p;
    while (i < length) {
        if (c[i] < i) {
            k = i % 2 && c[i];
            p = permutation[i];
            permutation[i] = permutation[k];
            permutation[k] = p;
            ++c[i];
            i = 1;
            result.push(permutation.slice());
        } else {
            c[i] = 0;
            ++i;
        }
    }
    return result;
};

/**
 * Returns the likeness of two sentences by testing the jaro-winkler
 * likeness of all permutations of the words in the sentence.
 *
 * @param string $a The first sentence to check
 * @param string $b The second sentence to check
 * @return float The nearest likeness of any permutation
 */
export let permutedSentenceJaroWinkler = (a, b) => {
    let highestMatch = 0;
    for (let aSentencePermutation of getPermutations(a.split(' '))) {
        for (let bSentencePermutation of getPermutations(b.split(' '))) {
            highestMatch = Math.max(
                highestMatch,
                jaroWinkler(
                    aSentencePermutation.join(' '),
                    bSentencePermutation.join(' ')
                )
            );
        }
    }
    return highestMatch;
};

/**
 * Round up to the nearest multiple
 *
 * @param numeric $a The number to round up
 * @param numeric $b The factor
 * @return float The rounded up nearest multiple
 */
export let ceilingTo = (a, b) => Math.ceil(a / b) * b;

/**
 * Round down to the nearest multiple
 *
 * @param numeric $a The number to round down
 * @param numeric $b The factor
 * @return float The rounded down nearest multiple
 */
export let floorTo = (a, b) => Math.floor(a / b) * b;

/**
 * Rounds to the nearest multiple
 *
 * @param numeric $a The number to round
 * @param numeric $b The factor
 * @return float The rounded nearest multiple
 */
export let roundTo = (a, b) => Math.round(a / b) * b;

/**
 * Rounds to the nearest factor of an integer
 *
 * @param numeric $a The number to round
 * @param int $b The factor of 10
 * @return int The rounded nearest multiple
 */
export let roundToFactor = (a, b) => roundTo(a, 10 ** b);

/**
 * Basically toFixed but keeps as a float
 *
 * @param numeric $a The number to round
 * @param int $b The numeric precision
 * @return float The number rounded to the nearest precision
 */
export let roundToPrecision = (a, b) => ((f) => Math.round(a * f) / f)(10 ** b);

/**
 * Clamps a number between a min and max.
 *
 * @param numeric $n The number to clamp.
 * @param numeric $min The minimum, default 0
 * @param numeric $max The maximum, default 1
 * @return numeric The number clamped between min and max
 */
export let clamp = (n, min=0, max=1) => Math.max(Math.min(isNaN(n) ? 0 : n, max), min);

/**
 * Given a CSS selector string, split it into constituent parts.
 *
 * @param string $selector The CSS selector to parse
 * @return object The parsed selector.
 */
export let parseSelector = (selector) => {
    let selectorParts = {};
    selector.split(/(?=\.)|(?=#)|(?=\[)/).forEach((token) => {
        switch (token[0]) {
            case '#':
                if (selectorParts.ids === undefined) selectorParts.ids = [];
                selectorParts.ids.push(token.slice(1));
                break;
            case '.':
                if (selectorParts.classes === undefined)
                    selectorParts.classes = [];
                selectorParts.classes.push(token.slice(1));
                break;
            case '[':
                if (selectorParts.attributes === undefined)
                    selectorParts.attributes = {};
                let [key, value] = token.slice(1, -1).split('=');
                selectorParts.attributes[key] = strip(value);
                break;
            default:
                if (selectorParts.tags === undefined) selectorParts.tags = [];
                selectorParts.tags.push(token);
                break;
        }
    });
    return selectorParts;
};

/**
 * Provides an iterator over an array in a shifting array size
 *
 * @param array $arr The input array
 * @param int $frameSize The size of the frame ot view
 * @return iterator<array>
 */
export function* shiftingFrameIterator(arr, frameSize) {
    for (let i = 0; i < arr.length - frameSize + 1; i++) {
        yield arr.slice(i, i + frameSize);
    }
}

/**
 * Given any number of arrays, zip the nth contents together.
 * Example: zip([1,1],[2,2]]) = [1,2], [1,2]
 *
 * @param array $arrays Any number of arrays.
 * @return array The zipped arrays.
 */
export function* zip(...arrays) {
    let minimumLength = Math.min(...arrays.map((arr) => arr.length));
    for (let i = 0; i < minimumLength; i++) {
        yield arrays.map((array) => array[i]);
    }
}

/**
 * Converts hue, saturation and lightness to red, green and blue.
 *
 * @param float $h The hue, 0-1
 * @param float $s The saturation, 0-1
 * @param float $l The lightness, 0-1
 * @return array<int> The values as RGB
 */
export let hslToRgb = (h, s, l) => {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        let hueToRgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };

        let q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        let p = 2 * l - q;
        r = hueToRgb(p, q, h + 1 / 3);
        g = hueToRgb(p, q, h);
        b = hueToRgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
};

/**
 * Converts red, green, and blue to hue, saturation, and lightness.
 *
 * @param int $r The red value, 0-255
 * @param int $g The green value, 0-255
 * @param int $b The blue value, 0-255
 * @return array<float> The values as HSL
 */
export let rgbToHsl = (r, g, b) => {
    r /= 255;
    g /= 255;
    b /= 255;
    let max = Math.max(r, g, b),
        min = Math.min(r, g, b),
        h,
        s,
        l = (max + min) / 2;

    if (max === min) {
        h = s = 0;
    } else {
        let d = max - min;
        s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
        switch (max) {
            case r:
                h = (g - b) / d + (g < b ? 6 : 0);
                break;
            case g:
                h = (b - r) / d + 2;
                break;
            case b:
                h = (r - g) / d + 4;
                break;
        }
        h /= 6;
    }

    return [h, s, l];
};

/**
 * Converts red, green, and blue to a hexadecimal string.
 *
 * @param int $r The red value, 0-255
 * @param int $g The green value, 0-255
 * @param int $b The blue value, 0-255
 * @return string The values as a hex color string
 */
export let rgbToHex = (r, g, b) => {
    let toHex = (c) => (c < 16 ? '0' : '') + c.toString(16);
    return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
};

/**
 * Converts hue, saturation and lightness to a hexadecimal string
 *
 * @param float $h The hue, 0-1
 * @param float $s The saturation, 0-1
 * @param float $l The lightness, 0-1
 * @return string The values as a hex color string
 */
export let hslToHex = (h, s, l) => {
    return rgbToHex(...hslToRgb(h, s, l));
};

/**
 * Converts a hexadecimal color string to RGB.
 *
 * @param string $hex The hex color string.
 * @return array<int> the color values as RGB
 */
export let hexToRgb = (hex) => {
    if (hex.startsWith('#')) {
        hex = hex.substr(1);
    }
    let r = parseInt(hex.substr(0, 2), 16),
        g = parseInt(hex.substr(2, 2), 16),
        b = parseInt(hex.substr(4), 16);
    return [r, g, b];
};

/**
 * Converts a hexadecimal color string to HSL.
 *
 * @param string $hex The hex color string.
 * @return array<float> the color values as HSL
 */
export let hexToHsl = (hex) => {
    return rgbToHsl(...hexToRgb(hex));
};

const pluralRules = new Intl.PluralRules('en-US', { type: 'ordinal' }),
    suffixes = new Map([
        ['one', 'st'],
        ['two', 'nd'],
        ['few', 'rd'],
        ['other', 'th']
    ]);

/**
 * Given a number, format the ordinal version of that number.
 * 1 = 1st, 2 = 2nd, 3 = 3rd, etc.
 *
 * @param int $n The number to format
 * @return string The formatted ordinal number
 */
export let formatOrdinal = (n) => `${n}${suffixes.get(pluralRules.select(n))}`;

/**
 * Prompt for one or more files, then resolve a promise with 
 * the passed fiule objects.
 *
 * @param string $contentType The content types to accept, like `image/*` or `text/html`.
 * @param bool $allowMultiple Whether or not to allow multiple selections. Default false.
 * @return Promise A promise that is resolved with the passed files.
 */
export let promptFiles = (contentType = "*", allowMultiple = false) => {
    return new Promise((resolve, reject) => {
        let inputElement = document.createElement("input"),
            onMouseMove = (e) => {
                window.removeEventListener("mousemove", onMouseMove);
                reject("No files selected.");
            };
        inputElement.type = "file";
        inputElement.multiple = allowMultiple;
        inputElement.accept = contentType;
        inputElement.onchange = () => {
            let inputFiles = Array.from(inputElement.files);
            window.removeEventListener("mousemove", onMouseMove);
            resolve(allowMultiple ? inputFiles : inputFiles[0]);
        };
        inputElement.click();
        window.addEventListener("mousemove", onMouseMove);
    });
};

/**
 * Strips HTML from a string, returning a formatted string.
 */
export let stripHTML = (text) => {
    let replaced = text.replace(/<[^>]*>?/gm, '');
    return replaced;
};

/**
 * Turns an HTML string into a set of document elements
 */
export let createElementsFromString = (text) => {
    let div = document.createElement("div");
    div.innerHTML = text.trim();
    return div.childNodes;
}
