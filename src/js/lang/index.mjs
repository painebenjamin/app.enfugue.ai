let Language = {};

Language.getString = function (stringCode) {
    let base = this.strings,
        current = base;

    for (let part of stringCode.split('.')) {
        if (!current.hasOwnProperty(part)) {
            current = undefined;
            break;
        }
        current = current[part];
    }

    if (current === undefined) {
        console.error("Couldn't find string code", stringCode);
        return '';
    }

    return current;
};

Language.strings = {};
Language.supportedLanguages = ['en'];

export { Language };
