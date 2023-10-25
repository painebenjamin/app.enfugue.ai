/** @module view/image */
import { View } from "./base.mjs";
import { waitFor, isEmpty } from "../base/helpers.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { PNG } from "../base/png.mjs";

const E = new ElementBuilder({
    "imageDetails": "enfugue-image-details",
    "imageBrowser": "enfugue-image-browser"
});

/**
 * The ImageView just provides a couple additional methods
 * on top of a basic Image.
 */
class ImageView extends View {
    /**
     * @var string The tag name, a basic image
     */
    static tagName = "img";

    /**
     * @param object $config The base config object
     * @param string $src The image source
     */
    constructor(config, src, usePng = true) {
        super(config);
        this.src = src;
        this.usePng = usePng;
        this.loadedCallbacks = [];
        this.metadata = {};
        if (!isEmpty(src)) {
            if (usePng) {
                let callable = PNG.fromURL;
                if (src instanceof File) {
                    callable = PNG.fromFile;
                } else if (src instanceof Image) {
                    src = src.src;
                }
                callable.call(PNG, src).then((png) => {
                    this.png = png;
                    this.metadata = {...this.metadata, ...png.metadata};
                    this.src = png.base64;
                    this.image = new Image();
                    this.image.onload = () => this.imageLoaded();
                    this.image.src = this.src;
                }).catch((e) => {
                    console.error(e);
                    this.loaded = true;
                    this.error = true;
                });
            } else {
                this.image = new Image();
                this.image.onload = () => this.imageLoaded();
                this.image.src = this.src;
            }
        }
    }

    /**
     * Sets the image source.
     *
     * @param string $src The new image source.
     */
    setImage(src) {
        if (this.src === src) {
            return;
        }
        this.loaded = false;
        if (this.usePng) {
            let callable = PNG.fromURL;
            if (src instanceof File) {
                callable = PNG.fromFile;
            } else if (src instanceof Image) {
                src = src.src;
            }
            callable.call(PNG, src).then((png) => {
                this.png = png;
                this.metadata = {...this.metadata, ...png.metadata};
                png.addMetadata(this.metadata);
                this.src = png.base64;
                this.image = new Image();
                this.image.onload = () => this.imageLoaded();
                this.image.src = this.src;
            }).catch((e) => {
                console.error(e);
                this.loaded = true;
                this.error = true;
            });
        } else {
            this.src = src;
            this.image = new Image();
            this.image.onload = () => this.imageLoaded();
            this.image.src = src;
        }
    }

    /**
     * Adds a callback to fire when the image loads
     *
     * @param callable $callback The function to call
     */
    onLoad(callback) {
        if (this.loaded) {
            callback(this);
        } else {
            this.loadedCallbacks.push(callback);
        }
    }

    /**
     * Waits for the loaded boolean to be set
     *
     * @return Promise
     */
    waitForLoad() {
        return waitFor(() => this.loaded);
    }

    /**
     * This fires onLoad callbacks.
     */
    imageLoaded() {
        this.loaded = true;
        this.width = this.image.width;
        this.height = this.image.height;
        for (let callback of this.loadedCallbacks) {
            callback(this);
        }
        this.loadedCallbacks = [];
        if (this.node !== undefined) {
            this.node.src(this.src);
        }
    }

    /**
     * Gets the data URL from the image.
     */
    getDataURL() {
        let canvas = document.createElement("canvas");
        canvas.width = this.width;
        canvas.height = this.height;
        let context = canvas.getContext("2d");
        context.drawImage(this.image, 0, 0);
        return canvas.toDataURL();
    }

    /**
     * Sets the image to the data version of itself
     */
    async getImageAsDataURL() {
        if (!this.src.startsWith("data")) {
            this.setImage(this.getDataURL());
            await this.waitForLoad();
        }
        return this.image;
    }

    /**
     * Gets the image data as a blob for saving
     *
     * @return Promise
     */
    async getBlob() {
        await this.waitForLoad();
        return this.png.blob;
    }

    /**
     * Halves the image dimensions
     */
    async downscale(ratio = 2) {
        await this.waitForLoad();
        let canvas = document.createElement("canvas")
        canvas.width = Math.floor(this.width / ratio);
        canvas.height = Math.floor(this.height / ratio);
        let context = canvas.getContext("2d");
        context.drawImage(this.image, 0, 0, canvas.width, canvas.height);
        this.setImage(canvas.toDataURL());
        await this.waitForLoad();
    }

    /**
     * Mirrors the image horizontally
     *
     * @return Promise
     */
    async mirrorHorizontally() {
        await this.waitForLoad();
        let canvas = document.createElement("canvas");
        canvas.width = this.width;
        canvas.height = this.height;
        let context = canvas.getContext("2d");
        context.translate(this.width, 0);
        context.scale(-1, 1);
        context.drawImage(this.image, 0, 0);
        this.setImage(canvas.toDataURL());
        await this.waitForLoad();
    }
    
    /**
     * Mirrors the image vertically
     *
     * @return Promise
     */
    async mirrorVertically() {
        await this.waitForLoad();
        let canvas = document.createElement("canvas");
        canvas.width = this.width;
        canvas.height = this.height;
        let context = canvas.getContext("2d");
        context.translate(0, this.height);
        context.scale(1, -1);
        context.drawImage(this.image, 0, 0);
        this.setImage(canvas.toDataURL());
        await this.waitForLoad();
    }

    /**
     * Rotates the image clockwise by 90 degrees
     */
    async rotateClockwise() {
        await this.waitForLoad();
        let canvas = document.createElement("canvas");
        canvas.width = this.height;
        canvas.height = this.width;
        let context = canvas.getContext("2d");
        context.translate(this.height, 0);
        context.rotate(Math.PI/2.0);
        context.drawImage(this.image, 0, 0);
        this.setImage(canvas.toDataURL());
        await this.waitForLoad();
    }
    
    /**
     * Rotates the image clockwise by 90 degrees
     */
    async rotateCounterClockwise() {
        await this.waitForLoad();
        let canvas = document.createElement("canvas");
        canvas.width = this.height;
        canvas.height = this.width;
        let context = canvas.getContext("2d");
        context.translate(0, this.width);
        context.rotate(3*Math.PI/2.0);
        context.drawImage(this.image, 0, 0);
        this.setImage(canvas.toDataURL());
        await this.waitForLoad();
    }

    /**
     * On build, simple set image source
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.src)) {
            await this.waitForLoad();
            node.attr("src", this.src);
        }
        return node;
    }

}

/**
 * The BackgroundImageVie uses a background image instead of an image tag.
 * This enables some additional fitting options.
 */
class BackgroundImageView extends ImageView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-background-image-view";
    
    /**
     * Sets the image source.
     *
     * @param string $src The new image source.
     */
    setImage(src) {
        if (this.src === src) {
            return;
        }
        this.loaded = false;
        this.onLoad(() => {
            if (this.node !== undefined) {
                this.node.find(".background").css("background-image", `url(${this.src})`);
            }
        });
        this.src = src;
        this.image = new Image();
        this.image.onload = () => this.imageLoaded();
        this.image.src = src;
    }

    /**
     * On build, set the background image.
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.src)) {
            let backgroundNode = E.div().class("background").css("background-image", `url(${this.src})`);
            node.append(backgroundNode);
        }
        return node;
    }
};

/**
 * The InspectorView shows some additional information about the image and shows
 * a navigable version of the image if the image is bigger than its container.
 */
class ImageInspectorView extends View {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-image-inspector";
    
    /**
     * @var string The class name for the loader
     */
    static loaderClassName = "loading-bar";

    /**
     * @var int The number of pixels on the edges to scroll images
     */
    static scrollPixels = 40;

    /**
     * @var int The number of milliseconds to wait inbetween scrolling
     */
    static scrollInterval = 50;

    /**
     * @var int The number of pixels scrolled per iteration (sorta)
     */
    static scrollFactor = 5;

    /**
     * @var int The number of pixels tall the image browser is
     */
    static imageBrowserHeight = 100;

    /**
     * @var float The rate of growth of the scroll rate
     */
    static scrollGrowRate = 0.1;

    /**
     * @param object $config The base config object
     * @param string $src The image source
     * @param string $name The name of the image to display
     * @param int    $width The width of the inspector
     * @param int    $height The height of the inspector
     */
    constructor(config, src, name, width, height) {
        super(config);
        this.src = src;
        this.name = name;
        this.width = width;
        this.height = height;
        this.imageView = new ImageView(config, src);

        this.imageView.onLoad(() => {
            if (this.node !== undefined) {
                this.node.find(E.getCustomTag("imageDetails")).content(this.subtitle);
                let imageBrowser = this.node.find(E.getCustomTag("imageBrowser"));
                if (this.imageView.width > this.width || this.imageView.height > this.height) {
                    imageBrowser.show();
                } else {
                    imageBrowser.hide();
                }
            }
        });
    }

    /**
     * Sets the inspector view's width/height
     */
    setDimension(width, height){
        this.width = width;
        this.height = height;
        if (this.node !== undefined) {
            this.node.css({
                "width": this.width,
                "height": this.height
            });
            if (this.imageView.loaded) {
                let imageBrowser = this.node.find(E.getCustomTag("imageBrowser"));
                if (this.imageView.width > this.width || this.imageView.height > this.height) {
                    this.updateIndicator();
                    imageBrowser.show();
                } else {
                    imageBrowser.hide();
                }
            }
        }
    }

    /**
     * Sets the image source
     *
     * @param string $src The image URL
     */
    setImage(src) {
        if (this.src === src) {
            return;
        }
        this.src = src;
        this.imageView.setImage(src);
    }

    /**
     * @return string The subtitle to display under the node
     */
    get details() {
        if (!isEmpty(this.imageView.width) && !isEmpty(this.imageView.height)) {
            return `${this.imageView.width}px × ${this.imageView.height}px`;
        }
        return null;
    }

    /**
     * Waits for the image to load
     *
     * @return Promise
     */
    waitForLoad() {
        return this.imageView.waitForLoad();
    }

    /**
     * The mouesenter handler for the image browser.
     */
    onBrowserMouseEnter(e) {
        this.inBrowser = true;
    }
    
    /**
     * The mouseleave handler for the image browser.
     */
    onBrowserMouseLeave(e) {
        this.inBrowser = false;
        this.isBrowsing = false;
    }
    
    /**
     * The mousemove handler for the image browser.
     */
    onBrowserMouseMove(e) {
        if (this.isBrowsing) {
            e.preventDefault();
            e.stopPropagation();
            this.checkMoveBrowser(
                e.offsetX,
                e.offsetY
            );
        }
    }
    
    /**
     * The mousedown handler for the image browser.
     */
    onBrowserMouseDown(e) {
        this.isBrowsing = true;
        e.preventDefault();
        e.stopPropagation();
        this.checkMoveBrowser(
            e.offsetX,
            e.offsetY
        );
    }
    
    /**
     * The mouseup handler for the image browser.
     */
    onBrowserMouseUp(e) {
        this.isBrowsing = false;
    }

    /**
     * The mouseenter handler for the overall node.
     */
    onNodeMouseEnter(e) {
        this.checkMousePosition(
            e.offsetX - this.node.element.childNodes[0].scrollLeft, 
            e.offsetY - this.node.element.childNodes[0].scrollTop
        );
    }

    /**
     * The mousemove handler for the overall node.
     */
    onNodeMouseMove(e) {
        this.checkMousePosition(
            e.offsetX - this.node.element.childNodes[0].scrollLeft, 
            e.offsetY - this.node.element.childNodes[0].scrollTop
        );
    }

    /**
     * The mouseleave handler for the overall node.
     */
    onNodeMouseLeave(e) {
        this.scrollX = 0;
        this.scrollY = 0;
        this.updateScrollNodes();
    }

    /**
     * Given an x and y coordinate, determine if any state should be changed.
     */
    checkMousePosition(x, y) {
        if (this.inBrowser) {
            this.scrollX = 0;
            this.scrollY = 0;
            this.updateScrollNodes();
            return;
        }

        if (x > (this.width - this.constructor.scrollPixels)) {
            if (this.scrollX < 1) {
                this.scrollX = 1;
            }
        } else if (x < this.constructor.scrollPixels) {
            if (this.scrollX > -1) {
                this.scrollX = -1;
            }
        } else {
            this.scrollX = 0;
        }
        
        if (y > (this.height - this.constructor.scrollPixels)) {
            if (this.scrollY < 1) {
                this.scrollY = 1;
            }
        } else if (y < this.constructor.scrollPixels) {
            if (this.scrollY > -1) {
                this.scrollY = -1;
            }
        } else {
            this.scrollY = 0;
        }
        this.updateScrollNodes();
        this.checkStartScroll();
    }

    /**
     * Given an X and Y position, determine if we should scroll
     */
    checkMoveBrowser(x, y) {
        let leftRatio = x / this.imageBrowserWidth,
            topRatio = y / this.imageBrowserHeight,
            imageMiddleX = leftRatio * this.imageView.width,
            imageMiddleY = topRatio * this.imageView.height,
            newScrollLeft = 0,
            newScrollTop = 0;

        if (imageMiddleX > (this.width / 2)) {
            newScrollLeft = (imageMiddleX - (this.width / 2));
        }
        
        if (imageMiddleY > (this.height / 2)) {
            newScrollTop = (imageMiddleY - (this.height / 2));
        }
        
        this.node.element.childNodes[0].scrollLeft = newScrollLeft;
        this.node.element.childNodes[0].scrollTop = newScrollTop;
        this.updateIndicator();
    }

    /**
     * Updates the left/up/down/right indicators in the DOM
     */
    updateScrollNodes() {
        let left = this.node.find(".left"),
            up = this.node.find(".up"),
            right = this.node.find(".right"),
            down = this.node.find(".down");

        if (this.scrollX < 0) {
            left.addClass("active");
        } else {
            left.removeClass("active");
        }
        
        if (this.scrollX > 0) {
            right.addClass("active");
        } else {
            right.removeClass("active");
        }
        
        if (this.scrollY < 0) {
            up.addClass("active");
        } else {
            up.removeClass("active");
        }
        
        if (this.scrollY > 0) {
            down.addClass("active");
        } else {
            down.removeClass("active");
        }
    }

    /**
     * If the user enters a scroll location, this will be triggered and start the interval
     */
    checkStartScroll() {
        if (isEmpty(this.scrollInterval)) {
            let lastScrollX, lastScrollY;
            this.scrollInterval = setInterval(() => {
                let isScrolling = false;
                
                if (this.scrollX != 0) {
                    isScrolling = true;
                    if (this.scrollX === lastScrollX) {
                        this.scrollX *= (1 + this.constructor.scrollGrowRate);
                    }
                    this.node.element.childNodes[0].scrollLeft = this.node.element.childNodes[0].scrollLeft + (this.scrollX * this.constructor.scrollFactor);
                    lastScrollX = this.scrollX;
                }

                if (this.scrollY != 0) {
                    isScrolling = true;
                    if (this.scrollY === lastScrollY) {
                        this.scrollY *= (1 + this.constructor.scrollGrowRate);
                    }
                    this.node.element.childNodes[0].scrollTop = this.node.element.childNodes[0].scrollTop + (this.scrollY * this.constructor.scrollFactor);
                    lastScrollY = this.scrollY;
                }

                if (!isScrolling) {
                    clearInterval(this.scrollInterval);
                    this.scrollInterval = null;
                } else {
                    this.updateIndicator();
                }

            }, this.constructor.scrollInterval);
        }
    }

    /**
     * @return int The ratio of height to visible height
     */
    get visibleHeightRatio() {
        return this.height / this.imageView.height;
    };

    /**
     * @return int The ratio of width to visible width
     */
    get visibleWidthRatio() {
        return this.width / this.imageView.width;
    };
    
    /**
     * @return int The static height of the image browser node
     */
    get imageBrowserHeight() {
        return this.constructor.imageBrowserHeight;
    };

    /**
     * @return int The calculated width of the image browser node
     */
    get imageBrowserWidth() {
        return this.imageView.width / (this.imageView.height / this.imageBrowserHeight);
    };
    
    /**
     * @return float The ratio the node is scrolled from left to right in [0, 1]
     */
    get scrollLeftRatio() {
        return this.node.element.childNodes[0].scrollLeft / (this.imageView.width - this.width);
    };

    /**
     * @return float The ratio the node is scrolled from top to bottom in [0, 1]
     */
    get scrollTopRatio() {
        return this.node.element.childNodes[0].scrollTop / (this.imageView.height - this.height);
    };

    /**
     * @return int The width of the indicator to place over the image browser
     */
    get indicatorWidth() {
        return this.imageBrowserWidth * this.visibleWidthRatio;
    };

    /**
     * @return int The height of the indicator to be placed over the image browser
     */
    get indicatorHeight() {
        return this.imageBrowserHeight * this.visibleHeightRatio;
    };
    
    /**
     * @return int The left position for the indicator based on scroll ratio
     */
    get indicatorLeft() {
        return this.scrollLeftRatio * (this.imageBrowserWidth - this.indicatorWidth);
    };

    /**
     * @return int The right position for the indicator based on scroll ratio
     */
    get indicatorTop() {
        return this.scrollTopRatio * (this.imageBrowserHeight - this.indicatorHeight);
    };

    /**
     * Updates the location of the indicator based on scroll ratio in the DOM
     */
    updateIndicator() {
        let indicator = this.node.find(E.getCustomTag("imageBrowser")).find("span");
        indicator.css({
            "width": this.indicatorWidth,
            "height": this.indicatorHeight,
            "left": this.indicatorLeft,
            "top": this.indicatorTop
        });
    }

    /**
     * On build, get the sub-view contents and the subtitle
     */
    async build() {
        let node = await super.build(),
            imageDetails = E.imageDetails().content(),
            imageIndicator = E.span(),
            imageBrowser = E.imageBrowser().content(
                E.img().src(this.src),
                imageIndicator
            ).on("mouseenter", (e) => this.onBrowserMouseEnter(e))
             .on("mouseleave", (e) => this.onBrowserMouseLeave(e))
             .on("mousemove", (e) => this.onBrowserMouseMove(e))
             .on("mousedown", (e) => this.onBrowserMouseDown(e))
             .on("mouseup", (e) => this.onBrowserMouseUp(e))
             .on("dblclick", (e) => { e.preventDefault(); e.stopPropagation(); });

        if (!this.imageView.loaded) {
            imageBrowser.hide();
            imageDetails.hide();
        } else {
            let scaledImageHeight = 100,
                scaledImageWidth = this.imageView.width / (this.imageView.height / 100),
                visibleHeightRatio = this.height / this.imageView.height,
                visibleWidthRatio = this.width / this.imageView.width,
                indicatorWidth = scaledImageWidth * visibleWidthRatio,
                indicatorHeight = scaledImageHeight * visibleHeightRatio;

            imageIndicator.css({
                "width": indicatorWidth,
                "height": indicatorHeight
            });
            imageDetails.content(this.subtitle);
        }

        node.content(
            E.div().class("image-view-container").content(await this.imageView.getNode()),
            E.div().class("up").content(E.i().class("fa-solid fa-angle-up")),
            E.div().class("left").content(E.i().class("fa-solid fa-angle-left")),
            E.div().class("right").content(E.i().class("fa-solid fa-angle-right")),
            E.div().class("down").content(E.i().class("fa-solid fa-angle-down")),
            imageDetails,
            imageBrowser
        ).css({
            "width": this.width,
            "height": this.height
        }).on("mouseenter", (e) => this.onNodeMouseEnter(e))
        .on("mousemove", (e) => this.onNodeMouseMove(e))
        .on("mouseleave", (e) => this.onNodeMouseLeave(e));

        return node;
    }
}

export { ImageView, BackgroundImageView, ImageInspectorView };
