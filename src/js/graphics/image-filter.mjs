/** @module graphics/image-filter.mjs */
import { sleep, waitFor } from "../base/helpers.mjs";

/**
 * Use no-operation as default filter
 */
function noop(image) {
    let pixel = image[this.thread.y][this.thread.x];
    this.color(pixel[0], pixel[1], pixel[2], pixel[3]);
}

/**
 * Provides a base class for GPU-accelerated image filters
 */
class ImageFilter {
    /**
     * @var callable The filter function
     */
    static filterFunction = noop;

    /**
     * @param string $source The source of the image.
     */
    constructor(source, execute = true) {
        this.source = source;
        this.image = new Image();
        this.executeOnLoad = execute;
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
        this.reset(this.executeOnLoad);
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
     * setConstants does nothing at root
     */
    setConstants(constants, execute = true) {
        if (execute) {
            this.execute();
        }
    }

    /**
     * Tests the filter function using a full transparent image.
     * @param int $imageHeight The height of the test image.
     * @param int $imageWidth The width of the test image.
     * @return array<array<int>> The result image after executing the filter function.
     */
    static testFilter(imageHeight = 10, imageWidth = 10) {
        let instance = new this();
        instance.reset(false);
        let constants = {...instance.constants},
            thread = {x: 0, y: 0},
            result = new Array(imageHeight).fill(null).map(() => { return new Array(imageWidth).fill(null); }),
            image = new Array(imageHeight).fill(null).map(() => { return new Array(imageWidth).fill(null).map(() => [Math.random(), Math.random(), Math.random(), 1.0]); }),
            color = (r, g, b, a) => result[thread.y][thread.x] = [r, g, b, a],
            state = {
                constants: constants,
                thread: thread,
                color: color
            };
        constants.width = imageWidth;
        constants.height = imageHeight;
        for (let i = 0; i < imageHeight; i++) {
            for (let j = 0; j < imageWidth; j++) {
                state.thread.y = i;
                state.thread.x = j;
                this.filterFunction.call(state, image);
            }
        }

        return result;
    }
}

/**
 * The callable executed by gpu.js for performing matrix convolution
 * Important values:
 *      image: a width × height matrix of floating-point RGBA values (i.e., image[n][m] = [0-1,0-1,0-1,0-1])
 *      this.thread.x: x dimension of the particular GPU thread.
 *      this.thread.y: y dimension of the particular GPU thread.
 *      this.constants.width: This width of the image.
 *      this.constants.height: This height of the image.
 *      this.constants.radius: The radius about (x, y) to convolve.
 *      this.constants.matrix: A normalized ((radius * 2) + 1)² matrix to multiply (adds up to 1)
 */
function matrixConvolution(image) {
    const pixelX = this.thread.x;
    const pixelY = this.thread.y;
    
    let rgb = [0.0, 0.0, 0.0];
    
    for (let i = -this.constants.radius; i <= this.constants.radius; i++) {
        for (let j = -this.constants.radius; j <= this.constants.radius; j++) {
            const sampleY = Math.min(Math.max(0, pixelY + i), this.constants.height - 1);
            const sampleX = Math.min(Math.max(0, pixelX + j), this.constants.width - 1);
            const pixelValue = image[sampleY][sampleX];
            const matrixValue = this.constants.matrix[i + this.constants.radius][j + this.constants.radius];

            rgb[0] = rgb[0] + (pixelValue[0] * matrixValue);
            rgb[1] = rgb[1] + (pixelValue[1] * matrixValue);
            rgb[2] = rgb[2] + (pixelValue[2] * matrixValue);
        }
    }
    
    this.color(rgb[0], rgb[1], rgb[2], 1.0);
}

/**
 * An extension of the ImageFilter that allows for M×M matrix functions
 * The matrix will be moved over the image and each pixel at (x, y) will 
 * set to the product of the matrix convolution over the pixel window centered
 * around (x, y) - this must always be an odd-sized matrix
 */
class MatrixImageFilter extends ImageFilter {
    /**
     * Set the filter function to the matrix convolution function
     */
    static filterFunction = matrixConvolution;

    /**
     * Gets the matrix
     * Default does nothing
     */
    getMatrix() {
        let matrix = new Array((this.constants.radius*2)+1).fill(null).map(() => {
            return new Array((this.constants.radius*2)+1).fill(0.0);
        });
        matrix[this.constants.radius][this.constants.radius] = 1.0;
        return matrix;
    }

    /**
     * Override parent reset to additionally set radius
     */
    reset(execute = true) {
        super.reset(false);
        this.constants.radius = 1;
        this.constants.matrix = this.getMatrix();
        if (execute) {
            this.execute();
        }
    }

    /**
     * Override setConstants to include radius
     */
    setConstants(constants, execute = true) {
        this.constants.radius = parseInt(constants.radius === undefined ? this.constants.radius : constants.radius);
        this.constants.matrix = this.getMatrix();
        if (execute) {
            this.execute();
        }
    }

    /**
     * Sets the new radius value and executes
     * @param int $newRadius The new radius, >= 1
     */
    set radius(newRadius) {
        this.constants.radius = parseInt(newRadius);
        this.constants.matrix = this.getMatrix();
        this.execute();
    }
}

class WeightedMatrixImageFilter extends MatrixImageFilter {
    /**
     * Override reset to include weight
     */
    reset(execute = true) {
        super.reset(false);
        this.constants.weight = 0;
        this.constants.matrix = this.getMatrix();
        if (execute) {
            this.execute();
        }
    }
    
    /**
     * Overrset setConstants to include weight
     */
    setConstants(constants, execute = true) {
        this.constants.weight = parseInt(constants.weight === undefined ? this.constants.weight : constants.weight);
        super.setConstants(constants, execute);
    }

    /**
     * Set weight and execute
     */
    set weight(newWeight) {
        this.constants.weight = parseInt(newWeight);
        this.constants.matrix = this.getMatrix();
        this.execute();
    }
}

export {
    ImageFilter,
    MatrixImageFilter,
    WeightedMatrixImageFilter
};
