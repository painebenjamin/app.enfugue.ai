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
    let topLeftY = this.thread.y - (this.thread.y % this.constants.size);
    let topLeftX = this.thread.x - (this.thread.x % this.constants.size);
    let pixel = image[topLeftY][topLeftX];
    this.color(pixel[0], pixel[1], pixel[2], pixel[3]);
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
        this.constants.size = 2;
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
