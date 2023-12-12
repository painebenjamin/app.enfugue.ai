/** @module controller/extras/01-caption-upsampler */
import { isEmpty, isEquivalent, sleep } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { View } from "../../view/base.mjs";
import { MenuController } from "../menu.mjs";
import { CaptionUpsampleFormView } from "../../forms/enfugue/prompts.mjs";

const E = new ElementBuilder({
    "message": "enfugue-message-view",
    "conversationStatus": "enfugue-conversation-status-view"
});

/**
 * Shows back-and-forth with the caption upsampler.
 */
class CaptionUpsamplerConversationView extends View {
    /**
     * @var int Milliseconds between status polls
     */
    static pollingInterval = 500;

    /**
     * @var string Custom HTML tag name
     */
    static tagName = "enfugue-conversation-view";

    /**
     * Construct with prompts and getter
     */
    constructor(config, prompts, getStatus) {
        super(config);
        this.prompts = prompts;
        this.promptIndex = 0;
        this.getStatus = getStatus;
        this.lastCaptions = [];
        this.lastTask = null;
    }

    /**
     * Call once to monitor conversation as it goes
     */
    async startMonitor() {
        while (true) {
            try {
                let currentStatus = await this.getStatus();
                if (["processing", "queued"].indexOf(currentStatus.status) === -1) {
                    this.setFinalStatus(currentStatus);
                    break;
                } else {
                    this.setIntermediateStatus(currentStatus);
                }
                await sleep(this.constructor.pollingInterval);
            } catch(e) {
                console.error(e);
                this.notify("error", "Invocation Error", `${e}`);
                break;
            }
        }
    }

    /**
     * Called during monitoring to indicate an intermediate status was received
     */
    async setIntermediateStatus(status) {
        if (!isEmpty(status.task)) {
            if (status.task !== this.lastTask) {
                this.node.find(E.getCustomTag("conversationStatus")).content(status.task);
                this.lastTask = status.task;
            }
        }
        if (!isEmpty(status.captions)) {
            if (!isEquivalent(this.lastCaptions, status.captions)) {
                let agentMessages = this.node.findAll(".agent");
                agentMessages[agentMessages.length-1].content(
                    E.div().content(status.captions[0])
                ).removeClass("loading");
                if (status.captions.length > 1) {
                    for (let i = 1; i < status.captions.length; i++) {
                        this.node.append(
                            E.message().class("agent").content(
                                E.div().content(status.captions[i])
                            )
                        );
                    }
                }
                this.lastCaptions = status.captions;
            }
        }
        if (!isEmpty(status.step)) {
            let resultsPerPrompt = status.total / this.prompts.length,
                promptIndex = Math.floor(status.step / resultsPerPrompt);
            if (promptIndex !== this.promptIndex) {
                this.node.append(
                    E.message().class("user").content(E.div().content(this.prompts[promptIndex])),
                    E.message().class("agent loading").content(E.div().content("&hellip;"))
                );
                this.promptIndex = promptIndex;
            }
        }
        let progress = ((isEmpty(status.progress) ? 0.0 : status.progress) * 100).toFixed(1);
        this.node.find(E.getCustomTag("conversationStatus")).css({
            "background-image": `linear-gradient(to right, var(--theme-color-primary) 0%, var(--theme-color-primary) calc(${progress}% - 1px), transparent ${progress}%`
        });
        this.node.render();
    }

    /**
     * Called during monitoring to indicate the final caption was received
     */
    async setFinalStatus(status) {
        let nodeContents = [];
        if (!isEmpty(status.result) && !isEmpty(status.result.captions)) {
            let promptIndex = 0;
            for (let promptResults of status.result.captions) {
                nodeContents.push(E.message().class("user").content(E.div().content(this.prompts[promptIndex++])));
                for (let promptResult of promptResults) {
                    nodeContents.push(E.message().class("agent").content(E.div().content(promptResult)));
                }
            }
            this.node.content(...nodeContents);
        } else if (status.status === "error" && !isEmpty(status.message)) {
            for (let loadingNode of this.node.findAll(".loading")) {
                this.node.remove(loadingNode);
            }
            this.node.append(E.message().class("agent").content(E.div().content(`ERROR: ${status.message}`)));
            this.node.render();
        }
    }

    /**
     * On build, append first caption
     */
    async build() {
        let node = await super.build();
        node.content(
            E.conversationStatus().content("Loading"),
            E.message().class("user").content(E.div().content(this.prompts[0])),
            E.message().class("agent loading").content(E.div().content("&hellip;"))
        );
        return node;
    }
}

/**
 * Shows the caption upsampler form and spawns conversations
 */
class CaptionUpsamplerController extends MenuController {
    /**
     * @var int width of the input window
     */
    static captionUpsampleWindowWidth = 400;

    /**
     * @var int height of the input window
     */
    static captionUpsampleWindowHeight = 460;

    /**
     * @var int width of the conversation window
     */
    static conversationWindowWidth = 400;

    /**
     * @var int height of the conversation window
     */
    static conversationWindowHeight = 300;

    /**
     * @var string The text in the UI
     */
    static menuName = "Caption Upsampler";
    
    /**
     * @var string The class of the icon in the UI
     */
    static menuIcon = "fa-solid fa-cloud-arrow-down";
    
    /**
     * @var string The keyboard shortcut
     */
    static menuShortcut = "c";

    /**
     * Show the new model form when clicked
     */
    async onClick() {
        this.showCaptionUpsampler();
    }

    /**
     * Builds the conversation view and starts the monitor
     */
    async startInvocationMonitor(prompts, result) {
        let conversationView = new CaptionUpsamplerConversationView(
            this.config,
            prompts,
            () => this.model.get(`/invocation/${result.uuid}`)
        );
        await this.spawnWindow(
            "Upsampler Conversation",
            conversationView,
            this.constructor.conversationWindowWidth,
            this.constructor.conversationWindowHeight,
        );
        conversationView.startMonitor();
    }

    /**
     * Shows the upsampler.
     * Creates if not yet done.
     */
    async showCaptionUpsampler() {
        if (!isEmpty(this.captionUpsamplerWindow)) {
            this.captionUpsampleWindow.focus();
            return;
        }
        let captionUpsampleForm = new CaptionUpsampleFormView(this.config);
        captionUpsampleForm.onSubmit(async (values) => {
            captionUpsampleForm.clearError();
            try {
                let result = await this.model.post("/invoke/language", null, null, values);
                this.startInvocationMonitor(values.prompts, result);
                captionUpsampleForm.enable();
            } catch(e) {
                captionUpsampleForm.setError(e);
                captionUpsampleForm.enable();
            }
        });
        this.captionUpsampleWindow = await this.spawnWindow(
            "Caption Upsampler",
            captionUpsampleForm,
            this.constructor.captionUpsampleWindowWidth,
            this.constructor.captionUpsampleWindowHeight
        );
    }
}

export { CaptionUpsamplerController as MenuController };
