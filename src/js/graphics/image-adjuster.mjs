/** NOCOMPRESS */
/** @module graphics/image-adjuster.mjs */
import { sleep, waitFor } from "../base/helpers.mjs";

/**
 * The callable that gpu.js will execute.
 * Important values are:
 *      image: a width × height matrix of floating-point RGBA values (i.e., image[n][m] = [0-1,0-1,0-1,0-1])
 *      this.thread.x: x dimension of the particular GPU thread.
 *      this.thread.y: y dimension of the particular GPU thread.
 *      this.constants.seed: A width × height matrix of CPU-generated random values from 0 to 1.
 *                           Note: this is one of the rare instances where GPU noise is not
 *                           sufficiently random; there are visible noise patterns when using it.
 *      this.constants.red: Red shift from -100 to 100.
 *      this.constants.green: Green shift from -100 to 100.
 *      this.constants.blue: Blue shift from -100 to 100.
 *      this.constants.hue: Hue shift from -100 to 100, where -100, and 100 are full hue wraparounds.
 *      this.constants.contrast: Contrast adjustment from -100 to 100
 *      this.constants.saturation: Saturation adjustment from -100 to 100
 *      this.constants.brightness: Naïve brightness adjustment from -100 to 100
 *      this.constants.lightness: Lightness enhancements from 0 to 100
 *      this.constants.noiseExponent: The number to factor the noise values by for exponential distribution
 *      this.constants.hueNoise: The amount of noise to add to the hue from 0 to 100
 *      this.constants.saturationNoise: The amount of noise to add to the saturation from 0 to 100
 *      this.constants.lightnessNoise: The amount of noise to add to the lightness from 0 to 100
 */
function performImageAdjustments(image) {
    // Get seed for this row
    const seed = this.constants.seed[this.thread.y][this.thread.x];
    
    // Convert red shift from [-100, 100] to [-1, 1]
    const redFactor = (this.constants.red / 100);

    // Convert green shift from [-100, 100] to [-1, 1]
    const greenFactor = (this.constants.green / 100);
    
    // Convert blue shift from [-100, 100] to [-1, 1]
    const blueFactor = (this.constants.blue / 100);
    
    // Convert contrast from [-100, 100] to [0, 2]
    const contrastFactor = (this.constants.contrast / 100) + 1;
    
    // Convert hue shift from [-100, 100] to [0, 1] using wraparound
    const hueFactor = (this.constants.hue < 0
        ? (100 + this.constants.hue)
        : this.constants.hue) / 100;

    // Convert saturation from [-100, 100] to [-1, 1]
    const saturationFactor = (this.constants.saturation / 100);

    // Convert brightness from [-100, 100] to [-1, 1]
    const brightnessFactor = (this.constants.brightness / 100);
    
    // Convert lightness from [0, 100] to [0, 1]
    const lightnessFactor = (this.constants.lightness / 100);

    // Convert hue noise from [0, 100] to [0, 1]
    const hueNoiseFactor = Math.pow((this.constants.hueNoise / 100), this.constants.noiseExponent);

    // Convert saturation noise from [0, 100] to [0, 1]
    const saturationNoiseFactor = Math.pow((this.constants.saturationNoise / 100), this.constants.noiseExponent);
    
    // Convert lightness noise from [0, 100] to [0, 1]
    const lightnessNoiseFactor = Math.pow((this.constants.lightnessNoise / 100), this.constants.noiseExponent);

    let pixel = image[this.thread.y][this.thread.x];

    // Adjust color levels
    pixel[0] = pixel[0] + (pixel[0] * redFactor);
    pixel[1] = pixel[1] + (pixel[1] * greenFactor);
    pixel[2] = pixel[2] + (pixel[2] * blueFactor);

    // Invert
    if (this.constants.invert === 1) {
        pixel[0] = 1.0 - pixel[0];
        pixel[1] = 1.0 - pixel[1];
        pixel[2] = 1.0 - pixel[2];
    }

    // Adjust Contrast
    pixel[0] = ((pixel[0] - 0.5) * contrastFactor) + 0.5;
    pixel[1] = ((pixel[1] - 0.5) * contrastFactor) + 0.5;
    pixel[2] = ((pixel[2] - 0.5) * contrastFactor) + 0.5;
    
    // Adjust Brightness
    pixel[0] = pixel[0] + brightnessFactor;
    pixel[1] = pixel[1] + brightnessFactor;
    pixel[2] = pixel[2] + brightnessFactor;

    // Clamp
    pixel[0] = Math.min(Math.max(pixel[0], 0.0), 1.0);
    pixel[1] = Math.min(Math.max(pixel[1], 0.0), 1.0);
    pixel[2] = Math.min(Math.max(pixel[2], 0.0), 1.0);

    // Convert to HSL
    const maxShade = Math.max(Math.max(pixel[0], pixel[1]), pixel[2]);
    const minShade = Math.min(Math.min(pixel[0], pixel[1]), pixel[2]);
    const difference = maxShade - minShade;
    
    let hue = 0.0;
    let saturation = 0.0;
    let lightness = (maxShade + minShade) / 2;

    if (maxShade != minShade) {
        saturation = lightness > 0.5
            ? difference / (2 - maxShade - minShade)
            : difference / (maxShade + minShade);
        if (maxShade == pixel[0]) {
            hue = (pixel[1] - pixel[2]) / difference + (pixel[1] < pixel[2] ? 6 : 0);
        } else if(maxShade == pixel[1]) {
            hue = (pixel[2] - pixel[0]) / difference + 2;
        } else if(maxShade == pixel[2]) {
            hue = (pixel[0] - pixel[1]) / difference + 4;
        }
        hue = hue / 6.0;
    }

    // Adjust hue
    hue = (hue + hueFactor) % 1.0;
    
    // Adjust saturation
    saturation = saturation + (saturation * saturationFactor);

    // Adjust lightness
    lightness = lightness + (lightness * lightnessFactor);
    
    // Add noise
    hue = (hue + ((0.5 - seed) * hueNoiseFactor)) % 1.0; // Wrap around
    saturation = saturation + ((0.5 - seed) * saturationNoiseFactor);
    lightness = lightness + ((0.5 - seed) * lightnessNoiseFactor);

    // Clamp
    saturation = Math.min(Math.max(saturation, 0.0), 1.0);
    lightness = Math.min(Math.max(lightness, 0.0), 1.0);

    // Convert back to floating-point RGB
    let red = 0;
    let green = 0;
    let blue = 0;

    if (saturation <= 0) {
        red = lightness;
        green = lightness;
        blue = lightness;
    } else {
        function hueToRGB(p, q, t) {
            if (t < 0) {
                t += 1;
            } else if (t > 1) {
                t -= 1;
            }
            if (t < 1/6) {
                return p + (q - p) * 6 * t;
            } else if (t < 1/2) {
                return q;
            } else if (t < 2/3) {
                return p + (q - p) * (2/3 - t) * 6;
            } else {
                return p;
            }
        }
        let q = lightness < 0.5 
            ? lightness * (1 + saturation) 
            : lightness + saturation - lightness * saturation;
        let p = 2 * lightness - q;
        red = hueToRGB(p, q, hue + 1/3);
        green = hueToRGB(p, q, hue);
        blue = hueToRGB(p, q, hue - 1/3);
    }
    
    // Finalize
    this.color(red, green, blue, pixel[3]);
}

/**
 * Provides a class that allows a number of adjustments to an image using GPU acceleration.
 * Usage is very simple. To adjust off-screen:
 *
 *      let adjuster = new ImageAdjuster("/images/myimage.png");
 *      adjuster.brightness = -25; // -25%
 *      adjuster.getImage().then((image) => document.body.appendChild(image));
 *
 * To adjust on-screen:
 * 
 *      let adjuster = new ImageAdjuster("/images/myimage.png");
 *      adjuster.getCanvas().then((canvas) => {
 *          document.body.appendChild(canvas);
 *          adjuster.brightness = -25; // -25%, will be visible show on screen
 *      });
 */
class ImageAdjuster {
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
        this.canvas.classList.add("adjuster");
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
        return this.canvas.getContext("webgl", {"preserveDrawingBuffer": true});
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
                const kernel = gpu.createKernel(performImageAdjustments)
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
     */
    reset() {
        this.constants = {
            "width": this.image.width,
            "height": this.image.height,
            "seed": this.seed,
            "noiseExponent": 3,
            "invert": 0,
            "contrast": 0,
            "brightness": 0,
            "lightness": 0,
            "saturation": 0,
            "hue": 0,
            "red": 0,
            "green": 0,
            "blue": 0,
            "hueNoise": 0,
            "saturationNoise": 0,
            "lightnessNoise": 0,
        };
        this.execute();
    }

    /**
     * Performs multiple adjustments at once
     *
     * @param object $adjustents The adjustment values
     * @see reset()
     */
    adjust(adjustments) {
        this.constants.contrast = parseInt(adjustments.contrast === undefined ? this.constants.contrast : adjustments.contrast);
        this.constants.brightness = parseInt(adjustments.brightness === undefined ? this.constants.brightness : adjustments.brightness);
        this.constants.saturation = parseInt(adjustments.saturation === undefined ? this.constants.saturation : adjustments.saturation);
        this.constants.lightness = parseInt(adjustments.lightness === undefined ? this.constants.lightness : adjustments.lightness);
        this.constants.hue = parseInt(adjustments.hue === undefined ? this.constants.hue : adjustments.hue);
        this.constants.red = parseInt(adjustments.red === undefined ? this.constants.red : adjustments.red);
        this.constants.green = parseInt(adjustments.green === undefined ? this.constants.green : adjustments.green);
        this.constants.blue = parseInt(adjustments.blue === undefined ? this.constants.blue : adjustments.blue);
        this.constants.hueNoise = parseInt(adjustments.hueNoise === undefined ? this.constants.hueNoise : adjustments.hueNoise);
        this.constants.saturationNoise = parseInt(adjustments.saturationNoise === undefined ? this.constants.saturationNoise : adjustments.saturationNoise);
        this.constants.lightnessNoise = parseInt(adjustments.lightnessNoise === undefined ? this.constants.lightnessNoise : adjustments.lightnessNoise);
        this.constants.invert = adjustments.invert === undefined ? this.constants.invert : adjustments.invert === true ? 1 : 0;
        this.execute();
    }

    /**
     * Sets the new contrast value and executes.
     * @param int $newContrast The new contrast between -100 and 100
     */
    set contrast(newContrast) {
        this.constants.contrast = parseInt(newContrast);
        this.execute();
    }
    
    /**
     * Sets the new brightness value and executes.
     *
     * @param int $newBrightness The new brightness between -100 and 100
     */
    set brightness(newBrightness) {
        this.constants.brightness = parseInt(newBrightness);
        this.execute();
    }

    /**
     * Sets the new lightness value and executes.
     *
     * @param int $newLightness The new lightness between 0 and 100
     */
    set lightness(newLightness) {
        this.constants.lightness = parseInt(newLightness);
        this.execute();
    }
    
    /**
     * Sets the new saturation value and executes.
     *
     * @param int $newSaturation The new saturation between -100 and 100
     */
    set saturation(newSaturation) {
        this.constants.saturation = parseInt(newSaturation);
        this.execute();
    }

    /**
     * Sets the new hue shift value and executes.
     *
     * @param int $newHue The new hue between -100 and 100
     */
    set hue(newHue) {
        this.constants.hue = parseInt(newHue);
        this.execute();
    }
    
    /**
     * Sets the new red shift value and executes.
     *
     * @param int $newRed The new red between -100 and 100
     */
    set red(newRed) {
        this.constants.red = parseInt(newRed);
        this.execute();
    }
    
    /**
     * Sets the new green value and executes.
     *
     * @param int $newGreen The new green shift value between -100 and 100
     */
    set green(newGreen) {
        this.constants.green = parseInt(newGreen);
        this.execute();
    }
    
    /**
     * Sets the new blue value and executes.
     *
     * @param int $newBlue The new blue shift between -100 and 100
     */
    set blue(newBlue) {
        this.constants.blue = parseInt(newBlue);
        this.execute();
    }

    /**
     * Sets the new hueNoise value and executes.
     *
     * @param int $newNoise The new hueNoise between -100 and 100
     */
    set hueNoise(newNoise) {
        this.constants.hueNoise = parseInt(newNoise);
        this.execute();
    }
    
    /**
     * Sets the new saturationNoise value and executes.
     *
     * @param int $newNoise The new saturationNoise between -100 and 100
     */
    set saturationNoise(newNoise) {
        this.constants.saturationNoise = parseInt(newNoise);
        this.execute();
    }
    
    /**
     * Sets the new lightnessNoise value and executes.
     *
     * @param int $newNoise The new lightnessNoise between -100 and 100
     */
    set lightnessNoise(newNoise) {
        this.constants.lightnessNoise = parseInt(newNoise);
        this.execute();
    }

    /**
     * Sets the new invert value and executes.
     *
     * @param bool $newInvert Whether or not to invert the image's colors.
     */
    set invert(newInvert) {
        this.constants.invert = newInvert === true ? 1 : 0;
        this.execute();
    }
}

export { ImageAdjuster };
