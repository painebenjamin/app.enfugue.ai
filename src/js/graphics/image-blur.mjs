/** NOCOMPRESS */
/** @module graphics/image-blur.mjs */
import { MatrixImageFilter } from "./image-filter.mjs";

/**
 * Provides a class that allows a gaussian blurring filter on an image.
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
};

/**
 * Provides a class that allows a box blurring filter on an image.
 */
class ImageBoxBlurFilter extends MatrixImageFilter {
    /**
     * Gets the gaussian distribution matrix
     */
    getMatrix() {
        let matrixSize = this.constants.radius * 2 + 1,
            matrixTotal = Math.pow(matrixSize, 2),
            matrix = new Array(matrixSize).fill(null).map(() => new Array(matrixSize).fill(1/matrixTotal));
        return matrix;
    }
}

export { 
    ImageGaussianBlurFilter,
    ImageBoxBlurFilter
};
