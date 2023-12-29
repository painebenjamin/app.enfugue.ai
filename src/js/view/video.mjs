/** @module view/video */
import { View } from "./base.mjs";
import { waitFor, isEmpty } from "../base/helpers.mjs";
import { ElementBuilder } from "../base/builder.mjs";

const E = new ElementBuilder();

/**
 * The VideoView mimics the capabilities of the ImageView
 */
class VideoView extends View {
    /**
     * @var string Tagname, we don't use view for a video
     */
    static tagName = "enfugue-video-view";

    /**
     * Construct with source
     */
    constructor(config, src) {
        super(config);
        this.loaded = false;
        this.loadedCallbacks = [];
        this.setVideo(src);
    }

    /**
     * Adds a callback to the list of loaded callbacks
     */
    onLoad(callback) {
        if (this.loaded) {
            callback(this);
        } else {
            this.loadedCallbacks.push(callback);
        }
    }

    /**
     * Wait for the video to be loaded
     */
    waitForLoad() {
        return waitFor(() => this.loaded);
    }

    /**
     * Sets the video source after initialization
     */
    setVideo(src) {
        if (this.src === src) {
            return;
        }
        this.loaded = false;
        this.src = src;
        this.video = document.createElement("video");
        this.video.onloadedmetadata = () => this.videoLoaded();
        this.video.autoplay = true;
        this.video.loop = true;
        this.video.muted = true;
        this.video.controls = true;
        this.video.src = src;
    }

    /**
     * Trigger video load callbacks
     */
    videoLoaded() {
        this.loaded = true;
        this.width = this.video.videoWidth;
        this.height = this.video.videoHeight;
        for (let callback of this.loadedCallbacks) {
            callback();
        }
    }

    /**
     * Sets the fit mode
     */
    setFit(newFitMode) {
        for (let fitMode of ["actual", "stretch", "cover", "contain"]) {
            let fitModeClass = `fit-${fitMode}`;
            if (fitMode === newFitMode) {
                this.addClass(fitModeClass);
            } else {
                this.removeClass(fitModeClass);
            }
        }
    }

    /**
     * Sets the anchor position
     */
    setAnchor(newAnchorMode, offsetX, offsetY) {
        let [topPart, leftPart] = newAnchorMode.split("-"),
            topPercent = topPart == "bottom" ? 100 : topPart == "center" ? 50 : 0,
            leftPercent = leftPart == "right" ? 100 : leftPart == "center" ? 50 : 0;

        this.css("object-position", `calc(${leftPercent}% + ${offsetX}px) calc(${topPercent}% + ${offsetY}px)`);
    }

    /**
     * Build the container and append the DOM node
     */
    async build() {
        let node = await super.build();
        node.content(this.video);
        return node;
    }
}

/**
 * An extension of the above that adds controls
 */
class VideoPlayerView extends VideoView {
    /**
     * After setting video, set controls
     */
    setVideo(src) {
        super.setVideo(src);
        this.video.controls = true;
    }
}

export {
    VideoView,
    VideoPlayerView
};
