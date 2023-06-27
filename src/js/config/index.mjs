let Configuration = {};

Configuration.debug = true;

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
        chunkingSize: 64,
        chunkingBlur: 64,
        guidanceScale: 7.5,
        inferenceSteps: 50,
        interval: 1000,
        upscaleDiffusionSteps: 80,
        upscaleDiffusionGuidanceScale: 10,
        upscaleDiffusionStrength: 0.15,
        upscaleDiffusionChunkingSize: 64,
        upscaleDiffusionChunkingBlur: 64,
        errors: {
            consecutive: 2
        }
    }
};

export { Configuration };
