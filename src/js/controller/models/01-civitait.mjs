/** @module controller/models/01-civitai */
import { isEmpty, humanSize, truncate, cleanHTML } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { View, ParentView, TabbedView } from "../../view/base.mjs";
import { SimpleNotification } from "../../common/notify.mjs";
import { CivitAISearchOptionsFormView } from "../../forms/enfugue/civitai.mjs";

const E = new ElementBuilder();
const browserWindowDimensions = [800, 1000];

/**
 * This is a single model from CivitAI (checkpoint, lora, TI)
 */
class CivitAIItemView extends View {
    /**
     * @var string The classname of the div
     */
    static className = "civit-ai-item";

    /**
     * @var int The maximum number of images to display for each item
     */
    static maxImagesPerItem = 3;

    /**
     * @var int The number of characters to show in the description before it is truncated.
     */
    static maxCharactersPerDescription = 300;

    /**
     * @var int The maximum number of tags to display per item
     */
    static maxTags = 6;

    constructor(config, item, download) {
        super(config);
        this.item = item;
        this.download = download;
    }

    /**
     * On build, assemble all the details and event handlers
     */
    async build() {
        let node = await super.build(),
            selectedVersion = this.item.modelVersions[0].name,
            name = E.h2().content(this.item.name),
            author = E.h4().content(`By ${this.item.creator.username}`),
            versionSelect = E.select(),
            versionContainer = E.div().class("versions"),
            flags = E.div().class("flags"),
            tags = E.div().class("tags"),
            description = E.p(),
            buildVersion = () => {
                let versionDetails = this.item.modelVersions.filter(
                        (version) => version.name === selectedVersion
                    ).shift(),
                    versionWords = versionDetails.trainedWords,
                    versionImages = versionDetails.images.slice(
                        0, 
                        this.constructor.maxImagesPerItem
                    ),
                    versionImageNodes = versionImages.map(
                        (image) => {
                            let node = E.img().src(image.url).css({
                                "max-width": `${((1/versionImages.length)*100).toFixed(2)}%`
                            });
                            if (!isEmpty(image.meta) && !isEmpty(image.meta.prompt)) {
                                if (!!navigator.clipboard) {
                                    // We can copy, bind clipboard write
                                    node.data("tooltip", `${cleanHTML(image.meta.prompt)}<br /><em class='note'>Ctrl+Right Click to copy prompt.</em>`);
                                    node.on("contextmenu", (e) => {
                                        if (e.ctrlKey) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            navigator.clipboard.writeText(image.meta.prompt);
                                            SimpleNotification.notify("Copied to Clipboard", 1000);
                                        }
                                    });
                                } else {
                                    node.data("tooltip", cleanHTML(image.meta.prompt));
                                }
                            }
                            return node;
                        }
                    ),
                    versionWordNodes = versionWords.map(
                        (word) => E.span().content(word)
                    ),
                    versionDownloads = versionDetails.files.map(
                        (file) => E.div().class("download").content(
                            E.span().class("name").content(file.name),
                            E.span().class("type").content(`${file.metadata.size || ""} ${file.metadata.fp || ""}`),
                            E.span().class("format").content(file.metadata.format),
                            E.span().class("size").content(humanSize(file.sizeKB * 1000)),
                            E.a().content(E.i().class("fa-solid fa-download")).data("tooltip", "Start Download").on("click", (e) => {
                                this.download(file.downloadUrl, file.name);
                            })
                        )
                    );

                versionContainer.content(
                    E.div().class("downloads").content(...versionDownloads),
                    E.div().class("images").content(...versionImageNodes),
                    E.div().class("triggers").content(...versionWordNodes)
                );
            };

        if (!isEmpty(this.item.description)) {
            if (this.item.description.length <= this.constructor.maxCharactersPerDescription) {
                description.content(this.item.description);
            } else {
                let truncatedDescription = E.span().content(truncate(this.item.description, this.constructor.maxCharactersPerDescription)),
                    fullDescription = E.span().content(this.item.description).hide(),
                    showFullDescription = E.a().content("Show All").on("click", () => { 
                        truncatedDescription.hide(); 
                        showFullDescription.hide();
                        fullDescription.show();
                    });

                description.content(truncatedDescription, showFullDescription, fullDescription);
            };
        }

        for (let tag of this.item.tags.slice(0, this.constructor.maxTags)) {
            tags.append(E.span().content(tag));
        }

        if (this.item.allowCommercialUse === "None") {
            flags.append(E.span().content("Commercial Use Disallowed"));
        } else if (this.item.allowCommercialUse === "Image") {
            flags.append(E.span().content("Commercial Use Allowed (Images Only)"));
        } else if (this.item.allowCommercialUse === "Rent" || this.item.allowCommercialUse === "Sell") {
            flags.append(E.span().content("Commercial Use Allowed"));
        } else {
            console.warning(`Unknown commercial use state '${this.item.allowCommercialUse}'`);
        }

        if (this.item.allowNoCredit) {
            flags.append(E.span().content("No Credit Required"));
        } else {
            flags.append(E.span().content("Credit Required"));
        }

        if (this.item.allowDerivatives) {
            flags.append(E.span().content("Derivative Models Allowed"));
        } else {
            flags.append(E.span().content("Derivative Models Disallowed"));
        }

        for (let version of this.item.modelVersions) {
            let option = E.option().content(`${version.name} (${version.baseModel})`);
            if (version.name === selectedVersion) {
                option.selected(true);
            }
            versionSelect.append(option);
        }

        versionSelect.on("change", (e) => {
            selectedVersion = versionSelect.val();
            buildVersion();
        });

        versionSelect.val(selectedVersion);
        buildVersion();

        node.append(name)
            .append(author)
            .append(flags)
            .append(tags)
            .append(description)
            .append(versionSelect)
            .append(versionContainer);
        return node;
    }
};


/**
 * This is solely the input for previous and next pages
 */
class CivitAICategoryPageView extends View {
    /**
     * @var string The div classname
     */
    static className = "page-buttons";

    constructor(config, showPrevious, showNext) {
        super(config);
        this.showPrevious = showPrevious;
        this.showNext = showNext;
        this.onPreviousCallbacks = [];
        this.onNextCallbacks = [];
    }

    /**
     * Add a callback triggered when the next button is clicked
     */
    onNext(callback) {
        this.onNextCallbacks.push(callback);
    }

    /**
     * Add a callback triggered when the previous button is clicked
     */
    onPrevious(callback){
        this.onPreviousCallbacks.push(callback);
    }

    /**
     * Trigger the previous callbacks
     */
    previousPage(){
        for (let callback of this.onPreviousCallbacks) {
            callback();
        }
    }
    
    /**
     * Trigger the next callbacks
     */
    nextPage(){
        for (let callback of this.onNextCallbacks) {
            callback();
        }
    }

    /**
     * On build, bind events and hide/show buttons as needed
     */
    async build() {
        let node = await super.build(),
            previous = E.button().content("Previous Page").on("click", () => this.previousPage()),
            next = E.button().content("Next Page").on("click", () => this.nextPage());

        if (!this.showPrevious) {
            previous.disabled(true);
        }

        if (!this.showNext) {
            next.disabled(true);
        }
        return node.content(previous, next);
    }
};

/**
 * The class holds items, input, and pagination
 */
class CivitAICategoryBrowserView extends ParentView {
    static inputTimeout = 500;
    static pageSize = 20;
    
    constructor(config, getData, download) {
        super(config);
        this.getData = getData;
        this.download = download;
        this.page = 1;
        this.query = "";
        this.timer = null;
        this.options = new CivitAISearchOptionsFormView(config);
        this.options.onSubmit(async (query) => {
            try {
                await this.runQuery(query);
            } catch(e) {
                SimpleNotification.notify("Couldn't communicate with Enfugue or CivitAI. Please try again.");
                console.error(e);
            }
            this.options.enable();
        });
        this.empty();
        this.addChild(this.options);
    }

    /**
     * Runs the query.
     * First empties itself and re-appends necessary items, then runs the query and
     * appends the new items in the child array.
     */
    async runQuery(query) {
        this.empty();
        this.addChild(this.options);
        
        let queryInput = {"page": this.page};
        if (!isEmpty(query.search)) {
            queryInput.query = query.search;
        }
        if (!isEmpty(query.sort)) {
            queryInput.sort = query.sort;
        }
        if (!isEmpty(query.period)) {
            queryInput.period = query.period;
        }
        if (query.commercial === true) {
            queryInput.allow_commercial_use = "Image";
        }
        if (query.nsfw === true) {
            queryInput.nsfw = true;
        }

        for (let datum of await this.getData(null, queryInput)) {
            let itemView = new CivitAIItemView(this.config, datum, (...args) => this.download(...args));
            await this.addChild(itemView);
        }

        let pager = new CivitAICategoryPageView(this.config, this.page > 1, this.children.length > this.constructor.pageSize);
        pager.onNext(async () => { 
            this.page++; 
            this.options.disable();
            await this.runQuery(query);
            this.options.enable();

        });
        pager.onPrevious(async () => {
            this.page--;
            this.options.disable();
            await this.runQuery(query);
            this.options.enable();
        });

        await this.addChild(pager);
    }

    /**
     * On build, append items we can get immediately, then trigger first lookup.
     */
    async build() {
        let node = await super.build();
        this.options.submit();
        return node;
    }
}

/**
 * This class holds the different categories, and adds the overall application callbacks
 * for when the user wants to download models
 */
class CivitAIBrowserView extends TabbedView {
    constructor(config, getCategoryData, download) {
        super(config);
        this.addTab(
            "Checkpoints", 
            new CivitAICategoryBrowserView(
                config, 
                (...args) => getCategoryData("checkpoint", ...args),
                (...args) => download("checkpoint", ...args)

            )
        );
        this.addTab(
            "LoRA", 
            new CivitAICategoryBrowserView(
                config, 
                (...args) => getCategoryData("lora", ...args),
                (...args) => download("lora", ...args)
            )
        );
        this.addTab(
            "LyCORIS", 
            new CivitAICategoryBrowserView(
                config, 
                (...args) => getCategoryData("lycoris", ...args),
                (...args) => download("lycoris", ...args)
            )
        );
        this.addTab(
            "Textual Inversion", 
            new CivitAICategoryBrowserView(
                config, 
                (...args) => getCategoryData("inversion", ...args),
                (...args) => download("inversion", ...args)
            )
        );
    }
};

/**
 * The overall view prepends the category browser with copyright information from CivitAI
 */
class CivitAIView extends View {
    /**
     * @var string The path to Civit AI's logo
     */
    static logoPath = "/static/img/brand/civit-ai-logo.svg";

    /**
     * @var string The class name for this overall view
     */
    static className = "civit-ai-view";

    constructor(config, getCategoryData, download) {
        super(config);
        this.browser = new CivitAIBrowserView(config, getCategoryData, download);
    }

    /**
     * @return DOMElement The description node.
     */
    getDescriptionNode() {
        return E.div().content(
            E.a().href("https://civitai.com").target("_blank").content(E.img().src(this.constructor.logoPath)),
            E.p().content("CivitAI is the leading AI model sharing service, providing a place where creators can uploads their trained models for other users to enjoy and employ."),
            E.p().content(
                E.span().content("By downloading any models from CivitAI, you agree to their "),
                E.a().href("https://civitai.com/content/tos").target("_blank").content("terms of service"),
                E.span().content(". Please review these terms before downloading anything. Some models come with additional terms, such as disallowing sale of images derived from those models. Use the filters at the top to only search for your desired terms, or pay attention to the tags below the model's name and author.")
            ),
            E.p().content(
                E.span().content("If you enjoy the service CivitAI provides, please consider "),
                E.a().href("https://civitai.com/pricing").target("_blank").content("subscribing or donating"),
                E.span().content(" to support free and open-source AI.")
            ),
            E.p().class("center").content(
                E.em().class("note").content("CivitAI, the CivitAI logo, and the CivitAI 'C' icon are all &copy; CivitAI " + (new Date()).getFullYear() + ", all rights reserved.")
            )
        );
    }

    /**
     * On build, prepend details
     */
    async build() {
        let node = await super.build();
        node.append(this.getDescriptionNode());
        node.append(await this.browser.getNode());
        return node;
    }
};

/**
 * The menu controller, will spawn the browser window.
 */
class CivitAIController extends MenuController {
    /**
     * @var string The text to display in the menu
     */
    static menuName = "Civit AI";

    /**
     * @var string The icon to display
     */
    static menuIcon = "/static/img/brand/civit-ai.png";

    /**
     * Shows the browser. Creates it if not yet made.
     */
    async showBrowser(){
        let getData = (category, ...args) => this.model.get(`civitai/${category}`, ...args),
            download = (category, url, filename) => this.download(category, url, filename);
        if (isEmpty(this.browser)) {
            this.browser = await this.spawnWindow(
                "CivitAI", 
                new CivitAIView(this.config, getData, download),
                ...browserWindowDimensions
            );
            this.browser.onClose(() => this.browser = null);
        } else {
            this.browser.focus();
        }
    }

    /**
     * The click handler to spawn the window.
     */
    async onClick() {
        this.showBrowser();
    }
}

export { CivitAIController as MenuController }
