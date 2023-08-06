/** @module graphics/image-filter.mjs */
import { sleep, waitFor } from "../base/helpers.mjs";

/**
 * Use no-operation as default filter
 */
function noop(image) {
    let pixel = image[this.thread.y][this.thread.x];
    this.color(pixel[0], pixel[1], pixel[2], pixel[3]);
}

class ImageFilter {
    /**
     * @var callable The filter function
     */
    static filterFunction = noop;

    /**
     * @param string $source The source of the image.
     */
    constructor(source) {
        this.source = source;
        this.image = new Image();
        this.image.onload = () => this.onload();
        this.image.src = source;
        this.loaded = false;
        this.constants = {};
    }

    /**
     * Called when the image has been loaded.
     */
    onload() {
        this.loaded = true;
        this.canvas = document.createElement("canvas");
        this.canvas.width = this.image.width;
        this.canvas.height = this.image.height;
        this.canvas.classList.add("image-filter");
        this.reset();
    }

    /**
     * Change the image source after instantiation.
     * 
     * @param string $newSource The new source of the image
     */
    setImage(newSource) {
        this.source = newSource
        this.loaded = false;
        this.image = new Image();
        this.image.onload = () => {
            this.canvas.width = this.image.width;
            this.canvas.height = this.image.height;
            this.constants.seed = this.seed; // Reload seed
            this.loaded = true;
            this.execute();
        }
        this.image.src = newSource;
    }

    /**
     * @return Promise A promise that will wait for the source image to be loaded.
     */
    awaitLoad() {
        return waitFor(() => this.loaded);
    }

    /**
     * @return CanvasRenderingContextWebGL The WebGL context for the canvas
     */
    get context() {
        if (this.canvasContext === null || this.canvasContext === undefined) {
            this.canvasContext = this.canvas.getContext("webgl", {"preserveDrawingBuffer": true});
        }
        return this.canvasContext;
    }
    
    /**
     * @return string The current result as a data URL
     */
    get imageSource() {
        return this.canvas.toDataURL("image/jpg");
    }

    /**
     * Compiles a kernel using the current image and settings.
     * 
     * @return Promise<GPU.Kernel> The callable kernel as a promise.
     */
    compileKernel() {
        return new Promise((resolve) => {
            this.getGPU().then((gpu) => {
                const kernel = gpu.createKernel(this.constructor.filterFunction)
                    .setConstants({...this.constants})
                    .setOutput([this.image.width, this.image.height])
                    .setGraphical(true);
                resolve(kernel);
            });
        });
    }

    /**
     * Compiles and executes the kernel.
     *
     * @return Promise A promise that resolves when complete.
     */
    execute() {
        return new Promise((resolve) => {
            this.compileKernel().then((kernel) => {
                kernel(this.image);
                resolve();
            });
        });
    }

    /**
     * Generate a matrix of random values of the same size as the image.
     *
     * @return array<array<float>>
     */
    get seed() {
        return (new Array(this.image.height).fill(null).map(
            () => (new Array(this.image.width).fill(null).map(
                () => Math.random()
            ))
        ));
    }
    
    /**
     * Gets a GPU instance, instantiates it if not yet done.
     * 
     * @return Promise<GPU.GPU>
     */
    getGPU() {
        if (this.gpu !== undefined) {
            return Promise.resolve(this.gpu);
        }
        return new Promise((resolve) => {
            this.awaitLoad().then(() => {
                this.gpu = new GPU.GPU({
                    "canvas": this.canvas,
                    "context": this.context
                });
                resolve(this.gpu);
            });
        });
    }

    /**
     * Gets the canvas that the GPU is rendering on.
     *
     * @return Promise<HTMLCanvas>
     */
    getCanvas() {
        return new Promise((resolve) => {
            this.awaitLoad().then(() => {
                resolve(this.canvas);
            });
        });
    }

    /**
     * Gets the current canvas as an image element.
     *
     * @return Promise<HTMLImage>
     */
    getImage() {
        return new Promise((resolve) => {
            this.awaitLoad().then(() => {
                let image = new Image();
                image.onload = () => { resolve(image); };
                image.src = this.imageSource;
            });
        });
    }
    
    /**
     * Resets constants to base values and executes.
     * @param bool $execute Whether or not to execute, default true
     */
    reset(execute = true) {
        this.constants = {
            "width": this.image.width,
            "height": this.image.height,
            "seed": this.seed,
        };
        if (execute) {
            this.execute();
        }
    }

    /**
     * Sets constants and then executes.
     *
     * @param object $constants New constants.
     * @param bool $execute Whether or not to execute, default true
     */
    setConstants(constants, execute = true) {
        this.constants = {
            ...this.constants,
            ...constants
        };
        if (execute) {
            this.execute();
        }
    }
}

export { ImageFilter };
