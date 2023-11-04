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
     * @var string Show canvas icon
     */
    static showCanvasIcon = "fa-solid fa-table-cells";

    /**
     * @var string Show canvas tooltip
     */
    static showCanvasTooltip = "Show the canvas, hiding any sample currently visible on-screen and revealing the grid and any nodes you've placed on it.";

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
     * @var string Tile vertical icon
     */
    static tileVerticalIcon = "fa-solid fa-ellipsis-vertical";

    /**
     * @var string Tile vertical tooltip
     */
    static tileVerticalTooltip = "Show the image tiled vertically.";

    /**
     * @var string Tile horizontal icon
     */
    static tileHorizontalIcon = "fa-solid fa-ellipsis";

    /**
     * @var string Tile horizontal tooltip
     */
    static tileHorizontalTooltip = "Show the image tiled horizontally.";

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
        this.showCanvasCallbacks = [];
        this.loopAnimationCallbacks = [];
        this.playAnimationCallbacks = [];
        this.tileHorizontalCallbacks = [];
        this.tileVerticalCallbacks = [];
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
     * Adds a callback to the show canvas button
     */
    onShowCanvas(callback){
        this.showCanvasCallbacks.push(callback);
    }

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
     * Adds a callback to the tile horizontal button
     */
    onTileHorizontal(callback) {
        this.tileHorizontalCallbacks.push(callback);
    }

    /**
     * Adds a callback to the tile vertical button
     */
    onTileVertical(callback) {
        this.tileVerticalCallbacks.push(callback);
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
     * Calls show canvas callbacks
     */
    showCanvas() {
        this.setActiveIndex(null);
        for (let callback of this.showCanvasCallbacks) {
            callback();
        }
    }

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
     * Calls tile horizontal callbacks
     */
    setHorizontalTile(tileHorizontal, updateDom = true) {
        for (let callback of this.tileHorizontalCallbacks) {
            callback(tileHorizontal);
        }
        if (!isEmpty(this.node) && updateDom) {
            let tileButton = this.node.find(".tile-horizontal");
            if (tileHorizontal) {
                tileButton.addClass("active");
            } else {
                tileButton.removeClass("active");
            }
        }
    }

    /**
     * Calls tile vertical callbacks
     */
    setVerticalTile(tileVertical, updateDom = true) {
        for (let callback of this.tileVerticalCallbacks) {
            callback(tileVertical);
        }
        if (!isEmpty(this.node) && updateDom) {
            let tileButton = this.node.find(".tile-vertical");
            if (tileVertical) {
                tileButton.addClass("active");
            } else {
                tileButton.removeClass("active");
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
            showCanvas = E.i()
                .addClass("show-canvas")
                .addClass(this.constructor.showCanvasIcon)
                .data("tooltip", this.constructor.showCanvasTooltip)
                .on("click", () => this.showCanvas()),
            tileHorizontal = E.i()
                .addClass("tile-horizontal")
                .addClass(this.constructor.tileHorizontalIcon)
                .data("tooltip", this.constructor.tileHorizontalTooltip)
                .on("click", () => {
                    tileHorizontal.toggleClass("active");
                    this.setHorizontalTile(tileHorizontal.hasClass("active"), false);
                }),
            tileVertical = E.i()
                .addClass("tile-vertical")
                .addClass(this.constructor.tileVerticalIcon)
                .data("tooltip", this.constructor.tileVerticalTooltip)
                .on("click", () => {
                    tileVertical.toggleClass("active");
                    this.setVerticalTile(tileVertical.hasClass("active"), false);
                }),
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

        samplesContainer
            .on("wheel", (e) => {
                e.preventDefault();
                samplesContainer.element.scrollLeft += e.deltaY / 10;
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
            showCanvas,
            E.div().class("tile-buttons").content(
                tileHorizontal,
                tileVertical
            ),
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
