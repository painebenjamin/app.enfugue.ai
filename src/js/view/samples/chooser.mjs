/** @module view/samples/chooser */
import { isEmpty, isEquivalent, bindMouseUntilRelease } from "../../base/helpers.mjs";
import { View } from "../base.mjs";
import { ImageView } from "../image.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { NumberInputView } from "../../forms/input.mjs";

const E = new ElementBuilder();

class SampleChooserView extends View {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-sample-chooser";

    /**
     * @var string Loop video icon
     */
    static loopIcon = "fa-solid fa-rotate-left";

    /**
     * @var string Loop video tooltip
     */
    static loopTooltip = "Loop the video, restarting it after it has completed.";
    
    /**
     * @var string Play video icon
     */
    static playIcon = "fa-solid fa-play";

    /**
     * @var string Play video tooltip
     */
    static playTooltip = "Play the animation.";

    /**
     * @var int default playback rate
     */
    static playbackRate = 8;

    /**
     * @var string playback rate tooltip
     */
    static playbackRateTooltip = "The playback rate of the animation in frames per second.";

    /**
     * @var string Text to show when there are no samples
     */
    static noSamplesLabel = "No samples yet. When you generate one or more images, their thumbnails will appear here.";

    /**
     * Constructor creates arrays for callbacks
     */
    constructor(config, samples = [], isAnimation = false) {
        super(config);
        this.loopAnimationCallbacks = [];
        this.playAnimationCallbacks = [];
        this.setActiveCallbacks = [];
        this.setPlaybackRateCallbacks = [];
        this.imageViews = [];
        this.isAnimation = isAnimation;
        this.samples = samples;
        this.activeIndex = 0;
        this.playbackRate = this.constructor.playbackRate;
        this.playbackRateInput = new NumberInputView(config, "playbackRate", {
            "min": 1,
            "max": 60,
            "value": this.constructor.playbackRate,
            "tooltip": this.constructor.playbackRateTooltip,
            "allowNull": false
        });
        this.playbackRateInput.onChange(
            () => this.setPlaybackRate(this.playbackRateInput.getValue(), false)
        );
    }

    // ADD CALLBACK FUNCTIONS

    /**
     * Adds a callback to the loop animation button
     */
    onLoopAnimation(callback) {
        this.loopAnimationCallbacks.push(callback);
    }

    /**
     * Adds a callback to the play animation button
     */
    onPlayAnimation(callback) {
        this.playAnimationCallbacks.push(callback);
    }

    /**
     * Adds a callback to when active is set
     */
    onSetActive(callback) {
        this.setActiveCallbacks.push(callback);
    }

    /**
     * Adds a callback to when playback rate is set
     */
    onSetPlaybackRate(callback) {
        this.setPlaybackRateCallbacks.push(callback);
    }

    // EXECUTE CALLBACK FUNCTIONS

    /**
     * Sets whether or not the samples should be controlled as an animation
     */
    setIsAnimation(isAnimation) {
        this.isAnimation = isAnimation;
        if (!isEmpty(this.node)) {
            if (isAnimation) {
                this.node.addClass("animation");
            } else {
                this.node.removeClass("animation");
            }
        }
    }

    /**
     * Calls loop animation callbacks
     */
    setLoopAnimation(loopAnimation, updateDom = true) {
        for (let callback of this.loopAnimationCallbacks) {
            callback(loopAnimation);
        }
        if (!isEmpty(this.node) && updateDom) {
            let loopButton = this.node.find(".loop");
            if (loopAnimation) {
                loopButton.addClass("active");
            } else {
                loopButton.removeClass("active");
            }
        }
    }

    /**
     * Calls play animation callbacks
     */
    setPlayAnimation(playAnimation, updateDom = true) {
        for (let callback of this.playAnimationCallbacks) {
            callback(playAnimation);
        }
        if (!isEmpty(this.node) && updateDom) {
            let playButton = this.node.find(".play");
            if (playAnimation) {
                playButton.addClass("active");
            } else {
                playButton.removeClass("active");
            }
        }
    }

    /**
     * Sets the active sample in the chooser
     */
    setActiveIndex(activeIndex, invokeCallbacks = true) {
        this.activeIndex = activeIndex;
        if (invokeCallbacks) {
            for (let callback of this.setActiveCallbacks) {
                callback(activeIndex);
            }
        }
        if (!isEmpty(this.imageViews)) {
            for (let i in this.imageViews) {
                let child = this.imageViews[i];
                if (i++ == activeIndex) {
                    child.addClass("active");
                } else {
                    child.removeClass("active");
                }
            }
        }
    }

    /**
     * Sets the playback rate
     */
    setPlaybackRate(playbackRate, updateDom = true) {
        this.playbackRate = playbackRate;
        for (let callback of this.setPlaybackRateCallbacks) {
            callback(playbackRate);
        }
        if (updateDom) {
            this.playbackRateInput.setValue(playbackRate, false);
        }
    }

    /**
     * Sets samples after initialization
     */
    async setSamples(samples) {
        let isChanged = !isEquivalent(this.samples, samples);
        this.samples = samples;

        if (!isEmpty(this.node)) {
            let samplesContainer = await this.node.find(".samples");
            if (isEmpty(this.samples)) {
                samplesContainer.content(
                    E.div().class("no-samples").content(this.constructor.noSamplesLabel)
                );
                this.imageViews = [];
            } else if (isChanged) {
                let samplesContainer = await this.node.find(".samples"),
                    render = false;

                if (isEmpty(this.imageViews)) {
                    samplesContainer.empty();
                    render = true;
                }

                for (let i in this.samples) {
                    let imageView,
                        imageViewNode,
                        sample = this.samples[i];

                    if (this.imageViews.length <= i) {
                        imageView = new ImageView(this.config, sample, false);
                        await imageView.waitForLoad();
                        imageViewNode = await imageView.getNode();
                        imageViewNode.on("click", () => {
                            this.setActiveIndex(i);
                        });
                        this.imageViews.push(imageView);
                        samplesContainer.append(imageViewNode);
                        render = true;
                    } else {
                        imageView = this.imageViews[i];
                        imageView.setImage(sample);
                        await imageView.waitForLoad();
                        imageViewNode = await imageView.getNode();
                    }

                    if (this.activeIndex !== null && this.activeIndex == i) {
                        imageView.addClass("active");
                    } else {
                        imageView.removeClass("active");
                    }

                    if (this.isAnimation) {
                        let widthPercentage = 100.0 / this.samples.length;
                        imageViewNode.css("width", `${widthPercentage}%`);
                    } else {
                        imageViewNode.css("width", null);
                    }
                }
                if (render) {
                    samplesContainer.render();
                }
            }
        }
    }

    /**
     * On build, add icons and selectors as needed
     */
    async build() {
        let node = await super.build(),
            loopAnimation = E.i()
                .addClass("loop")
                .addClass(this.constructor.loopIcon)
                .data("tooltip", this.constructor.loopTooltip)
                .on("click", () => {
                    loopAnimation.toggleClass("active");
                    this.setLoopAnimation(loopAnimation.hasClass("active"), false);
                }),
            playAnimation = E.i()
                .addClass("play")
                .addClass(this.constructor.playIcon)
                .data("tooltip", this.constructor.playTooltip)
                .on("click", () => {
                    playAnimation.toggleClass("active");
                    this.setPlayAnimation(playAnimation.hasClass("active"), false);
                }),
            samplesContainer = E.div().class("samples");

        let isScrubbing = false,
            getFrameIndexFromMousePosition = (e) => {
                let sampleContainerPosition = samplesContainer.element.getBoundingClientRect(),
                    clickRatio = e.clientX < sampleContainerPosition.left
                        ? 0
                        : e.clientX > sampleContainerPosition.left + sampleContainerPosition.width
                            ? 1
                            : (e.clientX - sampleContainerPosition.left) / sampleContainerPosition.width;

                return Math.min(
                    Math.floor(clickRatio * this.samples.length),
                    this.samples.length - 1
                );
            };

        let touchStart, touchScrollStart;
        samplesContainer
            .on("wheel", (e) => {
                e.preventDefault();
                samplesContainer.element.scrollLeft += e.deltaY / 10;
            })
            .on("touchstart", (e) => {
                touchStart = {x: e.touches[0].clientX, y: e.touches[0].clientY};
                touchScrollStart = samplesContainer.element.scrollLeft;
            })
            .on("touchmove", (e) => {
                let touchPosition = {x: e.touches[0].clientX, y: e.touches[0].clientY};
                if (isEmpty(touchStart)) {
                    touchStart = touchPosition;
                    touchScrollStart = samplesContainer.element.scrollLeft;
                } else {
                    let touchDelta = {
                        x: touchPosition.x - touchStart.x,
                        y: touchPosition.y - touchStart.y
                    };
                    samplesContainer.element.scrollLeft = touchScrollStart - touchDelta.x;
                }
            })
            .on("mousedown", (e) => {
                if (this.isAnimation) {
                    e.preventDefault();
                    e.stopPropagation();

                    isScrubbing = true;
                    this.setActiveIndex(getFrameIndexFromMousePosition(e));

                    bindMouseUntilRelease(
                        (e2) => {
                            if (isScrubbing) {
                                this.setActiveIndex(getFrameIndexFromMousePosition(e2));
                            }
                        },
                        (e2) => {
                            isScrubbing = false;
                        }
                    );
                }
            })
            .on("mousemove", (e) => {
                if (this.isAnimation) {
                    e.preventDefault();
                    e.stopPropagation();
                    if (isScrubbing) {
                        this.setActiveIndex(getFrameIndexFromMousePosition(e));
                    }
                }
            })
            .on("mouseup", (e) => {
                if (this.isAnimation) {
                    e.preventDefault();
                    e.stopPropagation();
                    isScrubbing = false;
                }
            });

        if (isEmpty(this.samples)) {
            samplesContainer.append(
                E.div().class("no-samples").content(this.constructor.noSamplesLabel)
            );
        } else {
            for (let i in this.samples) {
                let imageView,
                    imageViewNode,
                    sample = this.samples[i];

                if (this.imageViews.length <= i) {
                    imageView = new ImageView(this.config, sample, false);
                    imageViewNode = await imageView.getNode();
                    imageViewNode.on("click", () => {
                        this.setActiveIndex(i);
                    });
                    this.imageViews.push(imageView);
                } else {
                    imageView = this.imageViews[i];
                    imageView.setImage(sample);
                    imageViewNode = await imageView.getNode();
                }

                if (this.activeIndex !== null && this.activeIndex === i) {
                    imageView.addClass("active");
                } else {
                    imageView.removeClass("active");
                }

                samplesContainer.append(imageViewNode);
            }
        }

        node.content(
            samplesContainer,
            E.div().class("playback-rate").content(
                await this.playbackRateInput.getNode(),
                E.span().content("fps")
            ),
            loopAnimation,
            playAnimation
        );

        if (this.isAnimation) {
            node.addClass("animation");
        }

        return node;
    }
};

export { SampleChooserView };
