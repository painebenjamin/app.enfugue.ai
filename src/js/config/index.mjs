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
    "Automatic": {
        "themeColorPrimary": "#f36912",
        "themeColorSecondary": "#b1bacf",
        "themeColorTertiary": "#ffffff",
        "lightColor": "#dee2eb",
        "lighterColor": "#c7cedd",
        "lightestColor": "#ffffff",
        "darkColor": "#0b0f19",
        "darkerColor": "#1f2937",
        "darkestColor": "#06080f",
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
    "Retro": {
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
    "Plum": {
        "themeColorPrimary": "#673b80",
        "themeColorSecondary": "#c59eda",
        "themeColorTertiary": "#f6d8ee",
        "lightColor": "#f0d0f1",
        "lighterColor": "#f0e0f3",
        "lightestColor": "#f6effa",
        "darkColor": "#35243a",
        "darkerColor": "#1c1224",
        "darkestColor": "#0a070d",
        "headerFont": "Open Sans",
        "bodyFont": "Open Sans",
        "monospaceFont": "Open Sans"
    },
    "Greyscale": {
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
    },
    "Deep Dark": {
        "themeColorPrimary": "#303030",
        "themeColorSecondary": "#999999",
        "themeColorTertiary": "#999999",
        "lightColor": "#e1e1e1",
        "lighterColor": "#eaeaea",
        "lightestColor": "#f4f4f4",
        "darkColor": "#0b0b0b",
        "darkerColor": "#060606",
        "darkestColor": "#020202",
        "headerFont": "Roboto",
        "bodyFont": "Noto Sans",
        "monospaceFont": "Ubuntu Mono"
    },
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
        tilingStride: 128,
        tilingMaskType: "bilinear",
        guidanceScale: 6.5,
        inferenceSteps: 20,
        interval: 500,
        errors: {
            consecutive: 3
        }
    }
};

export { Configuration };
