/** NOCOMPRESS */
/** @module graphics/image-pixelizer.mjs */
import { ImageFilter } from "./image-filter.mjs";

/**
 * The callable that gpu.js will execute.
 * Important values are:
 *      image: a width × height matrix of floating-point RGBA values (i.e., image[n][m] = [0-1,0-1,0-1,0-1])
 *      this.thread.x: x dimension of the particular GPU thread.
 *      this.thread.y: y dimension of the particular GPU thread.
 *      this.constants.width: The width of the image.
 *      this.constants.height: The height of the image.
 *      this.constants.size: The size of the pixelizer.
 */
function performPixelize(image) {
    // Get top-left pixel position
    const topLeftY = this.thread.y - (this.thread.y % this.constants.size);
    const topLeftX = this.thread.x - (this.thread.x % this.constants.size);
    
    let pixel = [0.0, 0.0, 0.0];
    let sampleCount = 1;

    for (let i = 0; i < this.constants.size; i++) {
        for (let j = 0; j < this.constants.size; j++) {
            const sampleY = topLeftY + i;
            const sampleX = topLeftX + j;

            if (sampleY >= this.constants.height ||
                sampleX >= this.constants.width) {
                continue;
            }

            const pixelValue = image[sampleY][sampleX];
            pixel[0] = pixel[0] + pixelValue[0];
            pixel[1] = pixel[1] + pixelValue[1];
            pixel[2] = pixel[2] + pixelValue[2];
            sampleCount = sampleCount + 1;
        }
    }

    pixel[0] = pixel[0] / sampleCount;
    pixel[1] = pixel[1] / sampleCount;
    pixel[2] = pixel[2] / sampleCount;

    this.color(pixel[0], pixel[1], pixel[2], 1.0);
}

/**
 * Provides a class that allows a pixelizing filter on an image.
 * Usage is very simple. To filter off-screen:
 *
 *      let pixelizer = new ImagePixelizeFilter("/images/myimage.png");
 *      pixelizer.size = 10; // 10px
 *      pixelizer.getImage().then((image) => document.body.appendChild(image));
 *
 * To filter on-screen:
 * 
 *      let pixelizer = new ImagePixelizeFilter("/images/myimage.png");
 *      pixelizer.getCanvas().then((canvas) => {
 *          document.body.appendChild(canvas);
 *          pixelizer.size = 10; // 10px, will change on-screen
 *      });
 */
class ImagePixelizeFilter extends ImageFilter {
    /**
     * @var callable The filter function
     */
    static filterFunction = performPixelize;

    /**
     * Resets constants to base values and executes.
     */
    reset(execute = true) {
        super.reset(false);
        this.constants.size = 3;
        if (execute) {
            this.execute();
        }
    }

    /**
     * Updates size
     */
    setConstants(constants, execute = true) {
        this.constants.size = parseInt(constants.size === undefined ? this.constants.size : constants.size);
        if (execute) {
            this.execute();
        }
    }
    
    /**
     * Sets the new size value and executes.
     * @param int $newSize The new size, >= 2
     */
    set size(newSize) {
        this.constants.size = parseInt(newSize);
        this.execute();
    }
}

export { ImagePixelizeFilter };
