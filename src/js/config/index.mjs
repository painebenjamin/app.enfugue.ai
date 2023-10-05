let Configuration = {};

Configuration.debug = false;

Configuration.controller = {
    module: 'index'
};

Configuration.view = {
    applicationContainer: 'enfugue-application',
    language: 'en'
};

Configuration.url = {
    root: 'http://dev.enfugue.local/',
    api: 'http://dev.enfugue.local/api/',
    baseTitle: 'Enfugue',
    titleSeparator: ' | '
};

Configuration.history = {
    size: 100
};

Configuration.model = {
    cache: 0,
    cookie: {
        name: 'enfugue_token'
    },
    status: {
        interval: 10000,
    },
    autosave: {
        interval: 30000
    },
    queue: {
        interval: 5000
    },
    downloads: {
        interval: 5000
    },
    invocation: {
        width: 512,
        height: 512,
        engineSize: null,
        chunkingSize: 64,
        chunkingMaskType: "bilinear",
        guidanceScale: 7,
        inferenceSteps: 40,
        interval: 1000,
        errors: {
            consecutive: 2
        }
    }
};

export { Configuration };
