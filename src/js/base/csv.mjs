import { createEvent } from './helpers.mjs';

function escapeString(string) {
    return `"${string.replaceAll('"', '""')}"`;
}

function stringify(value) {
    return value instanceof Date
        ? value.toISOString().split('.')[0]
        : value.toString();
}

function checkEscapeString(string) {
    let needsEscaping = string.indexOf('"') > -1 || string.indexOf(',') > -1;
    return needsEscaping ? escapeString(string) : string;
}

function escapeValue(value) {
    return typeof value === 'string'
        ? checkEscapeString(value)
        : stringify(value);
}

class CSVFile {
    constructor(config) {
        this.config = config;
    }

    set data(data) {
        this.headers = Object.getOwnPropertyNames(data[0]);
        this.rows = data.map((datum) =>
            this.headers.map((header) => datum[header])
        );
    }

    get blob() {
        let headerString = this.headers.map((header) => escapeString(header)),
            rowStrings = this.rows.map((row) =>
                row.map((value) => escapeValue(value)).join(',')
            );

        return new Blob([[headerString].concat(rowStrings).join('\r\n')], {
            type: 'text/csv'
        });
    }

    get url() {
        if (this.objectUrl !== undefined) {
            window.URL.revokeObjectURL(this.objectUrl);
        }
        this.objectUrl = window.URL.createObjectURL(this.blob);
        return this.objectUrl;
    }

    download(name) {
        let link = document.createElement('a');
        link.setAttribute('download', name);
        link.href = this.url;
        link.innerText = 'Download';
        document.body.appendChild(link);
        window.requestAnimationFrame(() => {
            link.dispatchEvent(createEvent('click'));
            document.body.removeChild(link);
        });
    }
}

export { CSVFile };
