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
     * Build the container and append the DOM node
     */
    async build() {
        let node = await super.build();
        node.content(this.video);
        return node;
    }
}

class VideoPlayerView extends VideoView {
    setVideo(src) {
        super.setVideo(src);
        this.video.controls = true;
    }

    async build() {
        let node = await super.build();
        return node;
    }
}

export {
    VideoView,
    VideoPlayerView
};
