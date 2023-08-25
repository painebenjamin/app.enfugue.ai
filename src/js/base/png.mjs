/** @module base/png */
import { isEmpty } from "./helpers.mjs";

/**
 * Converts an array of bytes to a string
 */
function byteArrayToString(byteArray) {
    let str = "";
    for (let b of byteArray) {
        str += String.fromCharCode(b);
    }
    return str;
}

/**
 * Converts a string to an array of bytes
 */
function stringToByteArray(stringValue) {
    return (new Array(stringValue.length)).fill(null).map(
        (_, i) => stringValue.charCodeAt(i)
    );
}

/**
 * Converts a data URL into [mediaType, mediaFormat, dataFormat, data]
 */
function splitDataURL(url) {
    let [dataHeader, data] = url.split(","),
        [headerMedia, dataFormat] = dataHeader.split(";"),
        [_, mediaTypeFormat] = headerMedia.split(":"),
        [mediaType, mediaFormat] = mediaTypeFormat.split("/");

    return [mediaType, mediaFormat, dataFormat, data];
}

/**
 * Removes diacritics and other non-unicode characters from a string
 */
function normalize(stringValue) {
    return `${stringValue}`.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

/**
 * Static table for CRC calculations
 */
const CRCTable = new Int32Array(
    (new Array(256)).fill(null).map((_, i) => {
        let c = i;
        for (let j = 0; j < 8; j++) {
            c = ((c&1) ? (-306674912 ^ (c >>> 1)) : (c >>> 1));
        }
        return c;
    })
);

/**
 * Calculate CRC from a buffer of ints
 */
function CRC(buffer) {
    let chunkLength = buffer.length > 10000 ? 8 : 4,
        C = -1,
        L = buffer.length - (chunkLength - 1),
        i = 0;

    while (i < L) {
        for (let j = 0; j < chunkLength; j++) {
            C = (C>>>8) ^ CRCTable[(C^buffer[i++])&0xFF];
        }
    }
    while (i < L + (chunkLength - 1)) {
        C = (C>>>8) ^ CRCTable[(C^buffer[i++])&0xFF];
    }
    return C ^ -1;
};

/**
 * Memory-efficient and mostly speed-efficient type coercion
 */
const byteBuffer = new Uint8Array(4),
      int32ByteBuffer = new Int32Array(byteBuffer.buffer),
      uint32ByteBuffer = new Uint32Array(byteBuffer.buffer),
      bufferAsSigned = (newValue) => { 
        if (isEmpty(newValue)) {
            return int32ByteBuffer[0];
        } else {
            int32ByteBuffer[0] = newValue;
        }
      },
      bufferAsUnsigned = (newValue) => {
        if (isEmpty(newValue)) {
            return uint32ByteBuffer[0];
        } else {
            uint32ByteBuffer[0] = newValue;
        }
      };

/**
 * Provides ability to read and write PNG metadata in the frontend.
 */
class PNG {
    /**
     * @var array<int> the PNG header
     */
    static headerBytes = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /**
     * Pass the array buffer on construct.
     */
    constructor(name, buffer) {
        this.name = name;
        this.buffer = buffer;
    }

    /**
     * Gets the buffer as an array of bytes
     */
    get data() {
        return new Uint8Array(this.buffer);
    }

    /**
     * Reads a file into an array buffer and instantiates self
     */
    static fromFile(file) {
        return new Promise((resolve, reject) => {
            try {
                let reader = new FileReader();
                reader.onload = (e) => {
                    let png = new PNG(file.name, e.target.result);
                    try {
                        png.chunks;
                        resolve(png);
                    } catch(e) {
                        // If it failed here, it may not be a PNG. Coerce it.
                        let newReader = new FileReader();
                        newReader.onerror = (e) => reject(newReader);
                        newReader.onload = (e) => {
                            PNG.fromURL(e.target.result).then(resolve).catch(reject);
                        }
                        newReader.readAsDataURL(file);
                    }
                };
                reader.onerror = (e) => reject(reader);
                reader.readAsArrayBuffer(file);
            } catch(e) {
                reject(e);
            }
        });
    }

    /**
     * Instantiate a PNG from a base64 string
     * The string should _only_ include data, not the data:image/png;base64, part.
     */
    static fromBase64(name, encoded) {
        return new this(name, stringToByteArray(atob(encoded)));
    }

    /**
     * Instantiate a PNG from a URL to a PNG file
     */
    static fromImageURL(url) {
        let fileName = null;
        if (!url.startsWith("data")) {
            let [urlPart, queryPart] = url.split("?"),
                urlParts = urlPart.split("/");
            fileName = urlParts[urlParts.length-1];
        }
        return new Promise((resolve, reject) => {
            let xhr = new XMLHttpRequest();
            xhr.responseType = "blob";
            xhr.addEventListener("load", function(e) {
                if (this.status === 200) {
                    this.response.arrayBuffer().then((buffer) => {
                        resolve(new PNG(fileName, buffer));
                    });
                } else {
                    reject(this.response);
                }
            });
            xhr.open("GET", url);
            xhr.send();
        });
    }
    
    /**
     * Instantiate a PNG from a URL to any other kind of image file besides a PNG
     */
    static fromOtherImageURL(url) {
        let fileName = null;
        if (!url.startsWith("data")) {
            let [urlPart, queryPart] = url.split("?"),
                urlParts = urlPart.split("/");
            fileName = urlParts[urlParts.length-1];
        }
        return new Promise((resolve, reject) => {
            let image = new Image();
            image.onload = (e) => {
                let canvas = document.createElement("canvas");
                canvas.width = image.width;
                canvas.height = image.height;

                let context = canvas.getContext("2d");
                context.drawImage(image, 0, 0);

                let [mediaType, mediaFormat, dataFormat, data] = splitDataURL(canvas.toDataURL());
                resolve(this.fromBase64(fileName, data));
            }
            image.src = url;
        });
    }

    /**
     * The flexible entry-point for loading from URLs, determines which loader to use
     */
    static fromURL(url) {
        return new Promise((resolve, reject) => {
            if (url.startsWith("data")) {
                let [mediaType, mediaFormat, dataFormat, data] = splitDataURL(url);
                if (dataFormat === "base64" && mediaType === "image") {
                    if (mediaFormat === "png") {
                        resolve(this.fromBase64(null, data));
                    } else {
                        this.fromOtherImageURL(url).then(resolve).catch(reject);
                    }
                } else {
                    reject(`Bad data; must be base64 and an image. Got data format '${dataFormat}' and media type/format ${mediaType}/${mediaFormat}`);
                }
            } else {
                let [urlPart, queryPart] = url.split("?");
                if (urlPart.endsWith(".png")) {
                    this.fromImageURL(url).then(resolve).catch(reject);
                } else {
                    this.fromOtherImageURL(url).then(resolve).catch(reject);
                }
            }
        });
    }

    /**
     * Assembles a PNG from chunk array, see chunks() for details
     */
    static fromChunks(name, chunks) {
        let totalSize = this.headerBytes.length;

        for(let chunk of chunks) {
            totalSize += chunk.data.length + 12;
        }

        let output = new Uint8Array(totalSize),
            i = 0,
            writeIntoBuffer = (buffer, reverse) => {
                let length = buffer.length;
                for (let j = 0; j < length; j++) {
                    let chunkIndex = reverse
                        ? length - j - 1
                        : j;
                    output[i++] = buffer[chunkIndex];
                }
            };
        
        writeIntoBuffer(this.headerBytes, false);
        for (let chunk of chunks) {
            let nameBytes = stringToByteArray(chunk.name);
            bufferAsUnsigned(chunk.data.length);
            writeIntoBuffer(byteBuffer, true);
            writeIntoBuffer(nameBytes, false);
            writeIntoBuffer(chunk.data, false);
            bufferAsSigned(CRC(nameBytes.concat(Array.from(chunk.data))));
            writeIntoBuffer(byteBuffer, true);
        }

        return new PNG(name, output);
    }

    /**
     * Gets the image data as encoded chunks for manipulation
     */
    get chunks() {
        let dataArray = new Uint8Array(this.buffer),
            i = 0,
            chunks = [],
            ended = false,
            readIntoBuffer = (buffer, length, reverse = true, offset = 0) => {
                let initial = i;
                while (i - initial < length && i < dataArray.length) {
                    let chunkIndex = reverse
                        ? length - (i - initial) + offset - 1
                        : (i - initial) + offset;
                    buffer[chunkIndex] = dataArray[i++];
                }
            };

        for (; i < this.constructor.headerBytes.length; i++) {
            if (dataArray[i] !== this.constructor.headerBytes[i]) {
                throw `Invalid .png file header - expected ${this.constructor.headerBytes[i]} at index ${i}, but got ${dataArray[i]} instead.`;
            }
        }
        
        while (i < dataArray.length) {
            let chunkStart = i;
            // Chunk length
            readIntoBuffer(byteBuffer, 4);
            let chunkLength = bufferAsUnsigned() + 4,
                chunk = new Uint8Array(chunkLength);
            
            // Chunk name
            readIntoBuffer(chunk, 4, false);
            let chunkName = byteArrayToString(chunk.slice(0, 4));
            if (!chunks.length && chunkName !== "IHDR") {
                throw `First chunk does not contain IHDR (got ${chunkName}); malformed PNG file.`;
            }
            if (chunkName === "IEND") {
                ended = true;
                chunks.push({
                    name: chunkName,
                    data: new Uint8Array(0),
                    offset: chunkStart
                });
                break;
            }

            // Actual data
            readIntoBuffer(chunk, chunkLength - 4, false, 4);
            
            // CRC value
            readIntoBuffer(byteBuffer, 4);
            let actualCRC = bufferAsSigned(),
                expectedCRC = CRC(chunk);

            if (actualCRC !== expectedCRC) {
                throw `CRC Values are incorrect for ${chunkName} - expected ${expectedCRC}, got ${actualCRC}`;
            }

            chunks.push({
                name: chunkName,
                data: new Uint8Array(chunk.buffer.slice(4)),
                offset: chunkStart
            });
        }

        if (!ended) {
            throw "No IEND header found, malformed PNG file.";
        }

        return chunks;
    }

    /**
     * Encodes text data as expected by PNG tEXt chunks
     */
    encodeTextData(keyword, text) {
        let normalizedKeyword = normalize(keyword).substring(0, 79),
            normalizedText = normalize(text);

        return new Uint8Array(
            stringToByteArray(normalizedKeyword)
            .concat([0])
            .concat(stringToByteArray(normalizedText))
        );
    }

    /**
     * Decodes text data as formatted in PNG tEXt chunks
     */
    decodeTextData(textData) {
        let keywordIndex = textData.indexOf(0);
        if (keywordIndex === -1) {
            return {
                keyword: byteArrayToString(textData),
                text: ""
            };
        }
        return {
            keyword: byteArrayToString(textData.slice(0, keywordIndex)),
            text: byteArrayToString(textData.slice(keywordIndex + 1))
        };
    }

    /**
     * Adds a single metadatum and adjusts the buffer
     */
    addMetadatum(key, value) {
        return this.addMetadata({key: value});
    }

    /**
     * Adds an object fills with metadata and adjusts the buffer
     */
    addMetadata(metadata) {
        let encoded = Object.getOwnPropertyNames(metadata).reduce((carry, key) => {
                carry[key] = this.encodeTextData(key, `${metadata[key]}`);
                return carry;
            }, {}),
            chunks = this.chunks,
            chunkArray = [],
            chunkIndex = 0;

        for (let chunk of chunks) {
            if (chunk.name === "IDAT") {
                break;
            }
            if (chunk.name === "tEXt") {
                let decoded = this.decodeTextData(chunk.data);
                for (let key in encoded) {
                    if (key === decoded.key) {
                        chunk.data = encoded[key];
                        delete encoded[key];
                        break;
                    }
                }
            }
            chunkArray.push(chunk);
            chunkIndex++;
        }

        for (let key in encoded) {
            chunkArray.push({
                name: "tEXt",
                data: encoded[key]
            });
        }

        chunkArray = chunkArray.concat(chunks.slice(chunkIndex));
        this.buffer = PNG.fromChunks(this.name, chunkArray).buffer;
    }

    /**
     * Gets metadata chunks
     */
    get metadata() {
        let result = {};
        for (let chunk of this.chunks) {
            if (chunk.name === "tEXt") {
                let decoded = this.decodeTextData(chunk.data);
                result[decoded.keyword] = decoded.text;
            }
        }
        return result;
    }

    /**
     * Gets the PNG image as a base64 string
     */
    get base64() {
        return `data:image/png;base64,${btoa(byteArrayToString(this.data))}`;
    }

    /**
     * Gets the blob data that can be saved
     */
    get blob() {
        return new Blob([this.data], {"type": "image/png"});
    }

    /**
     * Gets an image using the base64 source string
     */
    get image() {
        let image = new Image();
        image.src = this.base64;
        return image;
    }
}

export { PNG };
