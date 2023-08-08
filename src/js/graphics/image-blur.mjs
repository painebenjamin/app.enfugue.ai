/** NOCOMPRESS */
/** @module graphics/image-blur.mjs */
import { MatrixImageFilter } from "./image-filter.mjs";

/**
 * Provides a class that allows a blurring filter on an image.
 * Usage is very simple. To filter off-screen:
 *
 *      let blurFilter = new ImageGaussianBlurFilter("/images/myimage.png");
 *      blurFilter.radius = 10; // 10px
 *      blurFilter.getImage().then((image) => document.body.appendChild(image));
 *
 * To filter on-screen:
 * 
 *      let blurFilter = new ImageGaussianBlurFilter("/images/myimage.png");
 *      blurFilter.getCanvas().then((canvas) => {
 *          document.body.appendChild(canvas);
 *          blurFilter.radius = 10; // 10px, will change on-screen
 *      });
 */
class ImageGaussianBlurFilter extends MatrixImageFilter {
    /**
     * Gets the gaussian distribution matrix
     */
    getMatrix() {
        let matrixSize = this.constants.radius * 2 + 1,
            matrix = new Array(matrixSize).fill(null).map(() => new Array(matrixSize).fill(null)),
            sigma = Math.max(1, this.constants.radius / 2),
            sigmaSquared = Math.pow(sigma, 2),
            sum = 0;
        
        for (let i = 0; i < matrixSize; i++) {
            for (let j = 0; j < matrixSize; j++) {
                let exponentNumerator = Math.pow(i - this.constants.radius, 2) + Math.pow(j - this.constants.radius, 2),
                    exponentDenominator = 2 * sigmaSquared,
                    eFactor = Math.pow(Math.E, -exponentNumerator / exponentDenominator),
                    matrixValue = eFactor / (2 * Math.PI * sigmaSquared);

                matrix[i][j] = matrixValue;
                sum += matrixValue;
            }
        }
        
        for (let i = 0; i < matrixSize; i++) {
            for (let j = 0; j < matrixSize; j++) {
                matrix[i][j] /= sum;
            }
        }

        return matrix;
    }
}

export { ImageGaussianBlurFilter };
