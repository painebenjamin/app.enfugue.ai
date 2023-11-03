/** @module nodes/image-editor/common */
import { ElementBuilder } from "../../base/builder.mjs";
import { View } from "../../view/base.mjs";

const E = new ElementBuilder();

/**
 * This class is a placeholder for an image
 */
class NoImageView extends View {
    /**
     * @var string The tag name
     */
    static tagName="enfugue-placeholder-view";

    /**
     * @var string The icon
     */
    static placeholderIcon = "fa-solid fa-link-slash";

    /**
     * @var string The text
     */
    static placeholderText = "No image, use the options menu to add one.";

    /**
     * On build, append text and icon
     */
    async getNode() {
        let node = await super.build();
        node.content(
            E.i().class(this.constructor.placeholderIcon),
            E.p().content(this.constructor.placeholderText)
        );
        return node;
    }
}

/**
 * This class is a placeholder for a video
 */
class NoVideoView extends NoImageView {
    /**
     * @var string The text
     */
    static placeholderText = "No video, use the options menu to add one.";
}

export {
    NoImageView,
    NoVideoView
}
