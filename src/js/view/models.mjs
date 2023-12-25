/** @module view/models */
import { isEmpty, cleanHTML } from "../base/helpers.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { View } from "./base.mjs";

const E = new ElementBuilder();

/**
 * Shows model metadata
 */
class ModelMetadataView extends View {
    /**
     * @var string Placeholder text
     */
    static placeholderText = "Model metadata loading&hellip;";

    /**
     * @var string Text to show when it cant be found
     */
    static noMetadataText = "Sorry, metadata could not be found for this model.";

    /**
     * @var string Mimic being civit AI item
     */
    static className = "civit-ai-item";

    /**
     * Sets the metadata once its loaded
     */
    setMetadata(metadata) {
        if (isEmpty(metadata)) {
            this.node.content(
                E.p().class("placeholder").content(this.constructor.noMetadataText)
            );
        } else {
            let description = cleanHTML(isEmpty(metadata.description) ? "" : metadata.description);
            if (isEmpty(description)) {
                description = "<em>No description provided.</em>";
            }
            this.node.empty();
            this.node.append(
                E.h2().content(
                    E.a().content(metadata.name)
                         .target("_blank")
                         .href(`https://civitai.com/models/${metadata.modelId}`)
                ),
                E.h4().content(`${metadata.model.name} (${metadata.baseModel} ${metadata.model.type})`),
                E.p().content(description)
            );
            if (!isEmpty(metadata.trainedWords)) {
                this.node.append(
                    E.div().class("triggers").content(...metadata.trainedWords.map(
                        (word) => E.span().content(word)
                    ))
                );
            }
            if (!isEmpty(metadata.images)) {
                let imagesToShow = metadata.images.slice(0, 4);
                this.node.append(
                    E.div().class("images").content(...imagesToShow.map((image) => {
                        let img;

                        if (image.type === "video") {
                            img = E.video().content(E.source().src(image.url)).autoplay(true).muted(true).loop(true).controls(false);
                        } else {
                            img = E.img().src(image.url);
                        }

                        if (!isEmpty(image.meta) && !isEmpty(image.meta.prompt)) {
                            img.data("tooltip", cleanHTML(image.meta.prompt));
                        }
                        img.css({
                            "max-width": `${100.0/imagesToShow.length}%`,
                            "max-height": "100%"
                        });
                        return img;
                    }))
                );
            }
            this.node.render();
        }
    }

    /**
     * On build, append placeholder
     */
    async build() {
        let node = await super.build();
        node.content(
            E.p().class("placeholder")
                 .content(this.constructor.placeholderText)
        );
        return node;
    }
}

export { ModelMetadataView };
