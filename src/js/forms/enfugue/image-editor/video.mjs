/** @module forms/enfugue/image-editor/video */
import { NumberInputView } from "../../../forms/input.mjs";
import { ImageEditorImageNodeOptionsFormView } from "./image.mjs";

class ImageEditorVideoNodeOptionsFormView extends ImageEditorImageNodeOptionsFormView {
    static fieldSets = {
        ...ImageEditorImageNodeOptionsFormView.fieldSets,
        ...{
            "Video Options": {
                "skipFrames": {
                    "label": "Skip Frames",
                    "class": NumberInputView,
                    "config": {
                        "min": 0,
                        "step": 1,
                        "value": 0,
                        "tooltip": "If set, this many frames will be skipped from the beginning of the video."
                    }
                },
                "divideFrames": {
                    "class": NumberInputView,
                    "label": "Divide Frames",
                    "config": {
                        "min": 1,
                        "step": 1,
                        "value": 1,
                        "tooltip": "If set, only the frames that are divided evenly by this number will be extracted. A value of 1 represents all frames being extracted. A value of 2 represents every other frame, 3 every third frame, etc."
                    }
                }
            }
        }
    }
};

export { ImageEditorVideoNodeOptionsFormView };
