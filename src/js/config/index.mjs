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

Configuration.theme = "ENFUGUE";
Configuration.themes = {
    "ENFUGUE": {
        "themeColorPrimary": "#FF3366",
        "themeColorSecondary": "#598392",
        "themeColorTertiary": "#AEC3B0",
        "lightColor": "#E7F1D0",
        "lighterColor": "#EFF3E0",
        "lightestColor": "#F7FAEF",
        "darkColor": "#2F2F2F",
        "darkerColor": "#141414",
        "darkestColor": "#0A0A0A",
        "headerFont": "Roboto",
        "bodyFont": "Noto Sans",
        "monospaceFont": "Ubuntu Mono"
    },
    "Ice": {
        "themeColorPrimary": "#066d72",
        "themeColorSecondary": "#0b7378",
        "themeColorTertiary": "#82b8b9",
        "lightColor": "#d0d9f1",
        "lighterColor": "#e0e4f3",
        "lightestColor": "#eff9fa",
        "darkColor": "#25292f",
        "darkerColor": "#161a20",
        "darkestColor": "#0e1014",
        "headerFont": "Raleway",
        "bodyFont": "Work Sans",
        "monospaceFont": "Inconsolata"
    },
    "Backwoods": {
        "themeColorPrimary": "#3A4D39",
        "themeColorSecondary": "#739072",
        "themeColorTertiary": "#ECE3CE",
        "lightColor": "#d3f1d0",
        "lighterColor": "#e5f3e0",
        "lightestColor": "#f0faef",
        "darkColor": "#222822",
        "darkerColor": "#101210",
        "darkestColor": "#030303",
        "headerFont": "Raleway",
        "bodyFont": "Noto Sans",
        "monospaceFont": "Ubuntu Mono"
    },
    "Analog Sunset": {
        "themeColorPrimary": "#CE5A67",
        "themeColorSecondary": "#F4BF96",
        "themeColorTertiary": "#FCF5ED",
        "lightColor": "#f7e6d2",
        "lighterColor": "#f6e0c8",
        "lightestColor": "#FCF5ED",
        "darkColor": "#1F1717",
        "darkerColor": "#151010",
        "darkestColor": "#0f0c0c",
        "headerFont": "Quicksand",
        "bodyFont": "Raleway",
        "monospaceFont": "Inconsolata"
    },
    "Jade": {
        "themeColorPrimary": "#2e9775",
        "themeColorSecondary": "#2e9775",
        "themeColorTertiary": "#2e9775",
        "lightColor": "#d0f1d5",
        "lighterColor": "#e0f3e4",
        "lightestColor": "#effaf3",
        "darkColor": "#191a1a",
        "darkerColor": "#121313",
        "darkestColor": "#0b0c0c",
        "headerFont": "Raleway",
        "bodyFont": "Poppins",
        "monospaceFont": "Inconsolata"
    },
    "Lithograph": {
        "themeColorPrimary": "#7c7c7c",
        "themeColorSecondary": "#d7d7d7",
        "themeColorTertiary": "#f1f1f1",
        "lightColor": "#e1e1e1",
        "lighterColor": "#dcdcdc",
        "lightestColor": "#f1f1f1",
        "darkColor": "#373737",
        "darkerColor": "#212121",
        "darkestColor": "#0d0d0d",
        "headerFont": "Quicksand",
        "bodyFont": "Open Sans",
        "monospaceFont": "Ubuntu Mono"
    }
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
        tilingSize: null,
        tilingStride: 64,
        tilingMaskType: "bilinear",
        guidanceScale: 6.5,
        inferenceSteps: 20,
        interval: 1000,
        errors: {
            consecutive: 2
        }
    }
};

export { Configuration };
