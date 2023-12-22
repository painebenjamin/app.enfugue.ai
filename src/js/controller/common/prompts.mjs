/** @module controllers/common/prompts */
import { isEmpty, bindMouseUntilRelease } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { View } from "../../view/base.mjs";
import { Controller } from "../base.mjs";
import { PromptTravelFormView } from "../../forms/enfugue/prompts.mjs";

const E = new ElementBuilder();

/**
 * This class represents a single prompt in the prompts track
 */
class PromptView extends View {
    /**
     * @var string css classname
     */
    static className = "prompt-view";

    /**
     * @var string classname of the edit icon
     */
    static editIcon = "fa-solid fa-edit";

    /**
     * @var string classname of the delete icon
     */
    static deleteIcon = "fa-solid fa-trash";

    /**
     * @var int Minimum number of milliseconds between ticks when dragging prompts
     */
    static minimumTickInterval = 100;

    /**
     * @var int Number of pixels from edges to drag
     */
    static edgeHandlerTolerance = 15;

    /**
     * @var int Number of pixels from edges to slide
     */
    static slideHandlerTolerance = 40;

    /**
     * @var int minimum number of frames per prompt
     */
    static minimumFrames = 2;

    constructor(config, total, positive = null, negative = null, start = null, end = null, weight = null) {
        super(config);
        this.total = total;
        this.positive = positive;
        this.negative = negative;
        if (isEmpty(start)) {
            this.start = 0;
        } else {
            this.start = start;
        }
        if (isEmpty(end)) {
            this.end = total;
        } else {
            this.end = end;
        }
        if (isEmpty(weight)) {
            this.weight = 1.0;
        } else {
            this.weight = weight;
        }
        this.showEditCallbacks = [];
        this.onRemoveCallbacks = [];
        this.onChangeCallbacks = [];
    }

    /**
     * Set the position of a node
     * Do not call this method directly
     */
    setPosition(node) {
        let startRatio, endRatio;
        if (isEmpty(this.start)) {
            startRatio = 0.0;
        } else {
            startRatio = this.start / this.total;
        }
        if (isEmpty(this.end)) {
            endRatio = 1.0;
        } else {
            endRatio = this.end / this.total;
        }
        node.css({
            "margin-left": `${startRatio*100.0}%`,
            "margin-right": `${(1.0-endRatio)*100.0}%`,
        });
    }

    /**
     * Resets the position of this node to where it should be
     */
    resetPosition() {
        if (!isEmpty(this.node)) {
            this.setPosition(this.node);
        }
    }

    /**
     * Adds a callback when the edit button is clicked
     */
    onShowEdit(callback) {
        this.showEditCallbacks.push(callback);
    }

    /**
     * Triggers showEdit callbacks
     */
    showEdit() {
        for (let callback of this.showEditCallbacks) {
            callback();
        }
    }

    /**
     * Adds a callback for when this prompt is removed
     */
    onRemove(callback) {
        this.onRemoveCallbacks.push(callback);
    }

    /**
     * Triggers remove callbacks
     */
    remove() {
        for (let callback of this.onRemoveCallbacks) {
            callback();
        }
    }

    /**
     * Adds a callback for when this is changed
     */
    onChange(callback) {
        this.onChangeCallbacks.push(callback);
    }

    /**
     * Triggers change callbacks
     */
    changed() {
        let state = this.getState();
        for (let callback of this.onChangeCallbacks) {
            callback(state);
        }
    }

    /**
     * Gets the state of this prompt from all sources
     */
    getState() {
        return {
            "positive": this.positive,
            "negative": this.negative,
            "weight": this.weight,
            "start": this.start,
            "end": this.end
        };
    }

    /**
     * Sets data that will be handled by an external form
     */
    setFormData(newData, zeroIndexed = true) {
        console.log(newData);
        console.trace();
        this.positive = newData.positive;
        this.negative = newData.negative;
        this.start = isEmpty(newData.start) ? this.start : newData.start - (zeroIndexed ? 0 : 1);
        this.end = isEmpty(newData.end) ? this.end : newData.end;
        this.weight = isEmpty(newData.weight) ? 1.0 : parseFloat(newData.weight);

        if (!isEmpty(this.node)) {
            let positive = this.node.find(".positive"),
                negative = this.node.find(".negative"),
                weight = this.node.find(".weight");

            weight.content(`${this.weight.toFixed(2)}`);

            if (isEmpty(this.positive)) {
                positive.content("(none)");
            } else {
                positive.content(this.positive);
                if (isEmpty(this.negative)) {
                    negative.hide();
                } else {
                    negative.show().content(this.negative);
                }
            }
        }

        this.resetPosition();
    }

    /**
     * Sets all state, form data and start/end
     */
    setState(newData) {
        if (isEmpty(newData)) newData = {};
        this.setFormData(newData);
    }

    /**
     * On build, append nodes for positive/negative, weight indicator and buttons
     */
    async build() {
        let node = await super.build(),
            weight = isEmpty(this.weight)
                ? 1.0
                : this.weight,
            edit = E.i().class(this.constructor.editIcon).on("click", () => {
                this.showEdit();
            }),
            remove = E.i().class(this.constructor.deleteIcon).on("click", () => {
                this.remove();
            }),
            positive = E.p().class("positive"),
            negative = E.p().class("negative"),
            prompts = E.div().class("prompts").content(positive, negative);

        if (isEmpty(this.positive)) {
            positive.content("(none)");
            negative.hide();
        } else {
            positive.content(this.positive);
            if (!isEmpty(this.negative)) {
                negative.content(this.negative);
            } else {
                negative.hide();
            }
        }

        let activeLeft = false,
            activeRight = false,
            activeSlide = false,
            canDragLeft = false,
            canDragRight = false,
            canSlide = false,
            lastTick = (new Date()).getTime(),
            slideStartFrame,
            slideStartRange,
            updateFrame = (closestFrame) => {
                if (activeLeft) {
                    this.start = Math.min(
                        closestFrame,
                        this.end - this.constructor.minimumFrames
                    );
                    this.setPosition(node);
                    this.changed();
                } else if(activeRight) {
                    this.end = Math.max(
                        closestFrame,
                        this.start + this.constructor.minimumFrames
                    );
                    this.setPosition(node);
                    this.changed();
                } else if(activeSlide) {
                    let difference = slideStartFrame - closestFrame,
                        [initialStart, initialEnd] = slideStartRange;

                    this.start = Math.max(initialStart - difference, 0);
                    this.end = Math.min(initialEnd - difference, this.total);
                    this.setPosition(node);
                    this.changed();
                }
            },
            updatePosition = (e) => {
                // e might be relative to window or to prompt container, so get absolute pos
                let now = (new Date()).getTime();
                if (now - lastTick < this.constructor.minimumTickInterval) return;
                lastTick = now;

                let promptPosition = node.element.getBoundingClientRect(),
                    promptWidth = promptPosition.width,
                    relativeLeft = Math.min(
                        Math.max(e.clientX - promptPosition.x, 0),
                        promptWidth
                    ),
                    relativeRight = Math.max(promptWidth - relativeLeft, 0),
                    containerPosition = node.element.parentElement.getBoundingClientRect(),
                    containerWidth = containerPosition.width - 15, // Padding
                    containerRelativeLeft = Math.min(
                        Math.max(e.clientX - containerPosition.x, 0),
                        containerWidth
                    ),
                    containerRelativeRight = Math.max(containerWidth - containerRelativeLeft, 0),
                    ratio = containerRelativeLeft / containerWidth,
                    closestFrame = Math.ceil(ratio * this.total);

                canDragLeft = false;
                canDragRight = false;
                canSlide = false;

                if (relativeLeft < this.constructor.edgeHandlerTolerance) {
                    node.css("cursor", "ew-resize");
                    canDragLeft = true;
                } else if (relativeRight < this.constructor.edgeHandlerTolerance) {
                    node.css("cursor", "ew-resize");
                    canDragRight = true;
                } else if (
                    relativeLeft >= this.constructor.slideHandlerTolerance &&
                    relativeRight >= this.constructor.slideHandlerTolerance
                ) {
                    node.css("cursor", "grab");
                    canSlide = true;
                    if (!activeSlide) {
                        slideStartFrame = closestFrame;
                    }
                } else if (!activeLeft && !activeRight && !activeSlide) {
                    node.css("cursor", "default");
                }

                updateFrame(closestFrame);
                e.preventDefault();
                e.stopPropagation();
            };

        node.content(
            prompts,
            E.div().class("weight").content(`${weight.toFixed(2)}`),
            edit,
            remove
        ).on("mouseenter", (e) => {
            updatePosition(e);
        }).on("mousemove", (e) => {
            updatePosition(e);
        }).on("mousedown", (e) => {
            if (e.which !== 1) return;
            if (canDragLeft) {
                activeLeft = true;
            } else if (canDragRight) {
                activeRight = true;
            } else if (canSlide) {
                activeSlide = true;
                slideStartRange = [this.start, this.end];
            }
            updatePosition(e);
            bindMouseUntilRelease(
                (e2) => { 
                    updatePosition(e2);
                },
                (e2) => { 
                    activeLeft = false;
                    activeRight = false;
                    activeSlide = false;
                }
            );
        }).on("dblclick", (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.showEdit();
        });
        this.setPosition(node);
        return node;
    }
}

/**
 * This view manages DOM interactions with the prompt travel sliders
 */
class PromptTravelView extends View {
    /**
     * @var string DOM tag name
     */
    static tagName = "enfugue-prompt-travel-view";

    /**
     * @var int Width of the prompt change form
     */
    static promptFormWindowWidth = 350;

    /**
     * @var int Height of the prompt change form
     */
    static promptFormWindowHeight = 500;

    /**
     * @var int Minimum number of frames
     */
    static minimumFrames = 2;

    /**
     * @var int The maximum number of divisions (notches)
     */
    static maximumDivisions = 64;

    /**
     * Constructor has callback to spawn a window
     */
    constructor(config, spawnWindow, length = 16) {
        super(config);
        this.length = length;
        this.spawnWindow = spawnWindow;
        this.promptViews = [];
        this.onChangeCallbacks = [];
    }

    /**
     * @return int how many frames per notch
     */
    get framesPerNotch() {
        let frameCount = this.length,
            framesPerNotch = 1;

        while (frameCount > this.constructor.maximumDivisions) {
            frameCount /= 2;
            framesPerNotch *= 2;
        }

        return framesPerNotch;
    }

    /**
     * @return int The count of frame notches
     */
    get frameNotches() {
        return Math.ceil(this.length / this.framesPerNotch);
    }

    /**
     * Sets the length of all frames, re-scaling notches and prompts
     */
    setLength(newLength) {
        this.length = newLength;
        let framesPerNotch = this.framesPerNotch,
            frameNotches = this.frameNotches;

        if (this.node !== undefined) {
            let notchNode = this.node.find(".notches"),
                newPromptNotches = new Array(frameNotches).fill(null).map(
                    (_, i) => E.span().class("notch").content(`${(i*framesPerNotch)+1}`)
                );

            notchNode.content(...newPromptNotches);
        }

        for (let promptView of this.promptViews) {
            promptView.total = this.length;
            promptView.end = Math.min(promptView.total, promptView.end);
            promptView.start = Math.max(0, Math.min(promptView.start, promptView.end-this.constructor.minimumFrames));
            promptView.resetPosition();
        }
    }

    /**
     * Adds a callback when any prompts are changed
     */
    onChange(callback) {
        this.onChangeCallbacks.push(callback);
    }

    /**
     * Triggers changed callbacks
     */
    changed() {
        let state = this.getState();
        for (let callback of this.onChangeCallbacks) {
            callback(state);
        }
    }

    /**
     * Removes a prompt view by pointer
     */
    removePrompt(promptView) {
        let promptViewIndex = this.promptViews.indexOf(promptView);
        if (promptViewIndex === -1) {
            console.error("Couldn't find prompt view in memory", promptView);
            return;
        }
        this.promptViews.splice(promptViewIndex, 1);
        if (!isEmpty(this.node) && !isEmpty(promptView.node)) {
            this.node.find(".prompts-container").remove(promptView.node);
        }
    }

    /**
     * Adds a new prompt from an option/state dictionary
     */
    async addPrompt(newPrompt) {
        if (isEmpty(newPrompt)) newPrompt = {};
        let promptView = new PromptView(
                this.config,
                this.length,
                newPrompt.positive,
                newPrompt.negative,
                newPrompt.start,
                newPrompt.end,
                newPrompt.weight
            ),
            promptFormView = new PromptTravelFormView(
                this.config,
                newPrompt
            ),
            promptWindow;

        promptFormView.onSubmit((values) => {
            promptView.setFormData(values, false);
            promptView.resetPosition();
            this.changed();
        });
        promptView.onChange(() => {
            this.changed();
            promptFormView.setValues({"start": promptView.start + 1, "end": promptView.end}, false);
        });
        promptView.onRemove(() => {
            this.removePrompt(promptView);
        });
        promptView.onShowEdit(async () => {
            if (!isEmpty(promptWindow)) {
                promptWindow.focus();
            } else {
                promptWindow = await this.spawnWindow(
                    "Edit Prompt",
                    promptFormView,
                    this.constructor.promptFormWindowWidth,
                    this.constructor.promptFormWindowHeight,
                );
                promptWindow.onClose(() => { promptWindow = null; });
            }
        });
        this.promptViews.push(promptView);
        if (this.node !== undefined) {
            this.node.find(".prompts-container").append(await promptView.getNode());
        }
    }

    /**
     * Empties the array of prompts in memory and DOM
     */
    emptyPrompts() {
        this.promptViews = [];
        if (!isEmpty(this.node)) {
            this.node.find(".prompts-container").empty();
        }
    }

    /**
     * Gets the state of all prompt views
     */
    getState() {
        return this.promptViews.map((view) => view.getState());
    }

    /**
     * Sets the state of all prompt views
     */
    async setState(newState = []) {
        this.emptyPrompts();
        for (let promptState of newState) {
            await this.addPrompt(promptState);
        }
        this.node.render();
    }

    /**
     * On build, add track and add prompt button
     */
    async build() {
        let node = await super.build(),
            framesPerNotch = this.framesPerNotch,
            frameNotches = this.frameNotches,
            addPrompt = E.button().content("Add Prompt").on("click", () => {
                this.addPrompt();
            }),
            promptNotches = new Array(frameNotches).fill(null).map(
                (_, i) => E.span().class("notch").content(`${(i*framesPerNotch)+1}`)
            ),
            promptNotchContainer = E.div().class("notches").content(...promptNotches),
            promptContainer = E.div().class("prompts-container"),
            promptsTrack = E.div().class("prompts-track").content(promptNotchContainer, promptContainer);

        for (let promptView of this.promptViews) {
            promptContainer.append(await promptView.getNode());
        }
        node.content(promptsTrack, addPrompt);
        return node;
    }
}

/**
 * The prompt travel controller is triggered by the prompt sidebar controller
 * It will show/hide the prompt travel view and manage state with the invocation engine
 */
class PromptTravelController extends Controller {
    /**
     * By default no prompts are provided
     */
    getDefaultState() {
        return {
            "travel": {
                "prompts": []
            }
        }
    }

    /**
     * We use the view class for easy state management
     */
    getState() {
        return {
            "travel": {
                "prompts": this.promptView.getState()
            }
        };
    }

    /**
     * Sets the state in the view class
     */
    async setState(newState) {
        if (!isEmpty(newState.travel)) {
            if (isEmpty(newState.travel.prompts)) {
                newState.travel.prompts = [];
            }
            await this.promptView.setState(newState.travel.prompts);
        }
    }

    /**
     * Disables prompt travel entirely (hides PT container)
     */
    async disablePromptTravel() {
        this.promptView.hide();
        this.application.container.classList.remove("prompt-travel");
        this.engine.prompts = null;
    }

    /**
     * Enables prompt travel
     * If there was previous state, keeps that. If there wasn't, add the current satte of the main prompts
     */
    async enablePromptTravel() {
        let currentState = this.promptView.getState();
        if (currentState.length === 0) {
            // Add the current prompt
            let positive = [this.engine.prompt, this.engine.prompt2];
            let negative = [this.engine.negativePrompt, this.engine.negativePrompt2];

            positive = positive.filter((value) => !isEmpty(value));
            if (isEmpty(positive)) {
                positive = null;
            } else if(positive.length === 1) {
                positive = positive[0];
            }

            negative = negative.filter((value) => !isEmpty(value));
            if (isEmpty(negative)) {
                negative = null;
            } else if(negative.length === 1) {
                negative = negative[0];
            }

            this.promptView.addPrompt({
                positive: positive,
                negative: negative
            });
        }
        this.promptView.show();
        this.application.container.classList.add("prompt-travel");
        this.engine.prompts = this.promptView.getState();
    }

    /**
     * On initialize, append and hide prompt travel view, waiting for the enable
     */
    async initialize() {
        this.promptView = new PromptTravelView(
            this.config,
            (title, content, w, h, x, y) => this.spawnWindow(title, content, w, h, x, y)
        );
        this.promptView.hide();
        this.promptView.onChange((newPrompts) => {
            this.engine.prompts = newPrompts;
        });
        this.application.container.appendChild(await this.promptView.render());
        this.subscribe("promptTravelEnabled", () => this.enablePromptTravel());
        this.subscribe("promptTravelDisabled", () => this.disablePromptTravel());

        // Let the sidebar cascade enable and disable events; we'll just change size
        this.subscribe("engineAnimationFramesChange", (frames) => {
            if (!isEmpty(frames) && frames > 0) {
                this.promptView.setLength(frames);
            }
        });
    }
}

export { PromptTravelController };
