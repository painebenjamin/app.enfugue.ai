import { FormView } from "../../view/forms/base.mjs";
import { 
    TextInputView,
    SelectInputView,
    FloatInputView,
    NumberInputView,
    FileInputView,
} from "../../view/forms/input.mjs";

const invocationFieldSets = {
    "Basics": {
        "model": {
            "label": "Model",
            "class": SelectInputView,
            "config": {
                "required": true
            }
        },
     },
    "Prompts": {
        "prompt": {
            "label": "Prompt",
            "class": TextInputView,
            "config": {
                "required": true
            }
        },
        "negative_prompt": {
            "label": "Negative Prompt",
            "class": TextInputView
        }
    },
    "Tweaks": {
        "seed": {
            "label": "Seed",
            "class": NumberInputView
        },
        "guidance_scale": {
            "label": "Guidance Scale",
            "class": FloatInputView,
            "config": {
                "required": false,
                "min": 0.0,
                "max": 10.0,
                "value": 7.5,
                "step": 0.1
            }
        },
        "inference_steps": {
            "label": "Inference Steps",
            "class": NumberInputView,
            "config": {
                "required": false,
                "min": 5,
                "max": 250,
                "value": 50
            }
        }
    },
    "Options": {
        "samples": {
            "label": "Samples",
            "class": NumberInputView,
            "config": {
                "required": false,
                "min": 1,
                "value": 1,
                "max": 4
            }
        },
        "iterations": {
            "label": "Iterations",
            "class": NumberInputView,
            "config": {
                "required": false,
                "min": 1,
                "value": 1,
                "max": 25
            }
        }
    }
}


class Txt2ImgFormView extends FormView {
    static fieldSets = {
        ...{
            "Dimensions": {
                "width": {
                    "label": "Width",
                    "class": NumberInputView,
                    "config": {
                        "required": true,
                        "min": 512,
                        "max": 4096,
                        "step": 8,
                        "value": 512
                    }
                },
                "height": {
                    "label": "Height",
                    "class": NumberInputView,
                    "config": {
                        "required": true,
                        "min": 512,
                        "max": 4096,
                        "step": 8,
                        "value": 512
                    }
                }
            }
        },
        ...invocationFieldSets
    };
};

class Img2ImgFormView extends FormView {
    static fieldSets = {
        ...{
            "Image": {
                "image": {
                    "label": "Image",
                    "class": FileInputView,
                    "config": {
                        "required": true
                    }
                },
                "strength": {
                    "label": "Strength",
                    "class": FloatInputView,
                    "config": {
                        "required": false,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "value": 0.8
                    }
                }
            }
        },
        ...invocationFieldSets
    };
};

export {
    Txt2ImgFormView,
    Img2ImgFormView
};
