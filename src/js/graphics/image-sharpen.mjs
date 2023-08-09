/** NOCOMPRESS */
/** @module graphics/image-blur.mjs */
import { WeightedMatrixImageFilter } from "./image-filter.mjs";

/**
 * Provides a class that allows a simple sharpen filter on an image.
 */
class ImageSharpenFilter extends WeightedMatrixImageFilter {
    /**
     * Gets the gaussian distribution matrix
     */
    getMatrix() {
        let matrixSize = this.constants.radius * 2 + 1,
            matrix = new Array(matrixSize).fill(null).map(() => new Array(matrixSize).fill(0)),
            matrixWeight = this.constants.weight / 100.0;
            
        for (let i = 0; i < matrixSize; i++) {
            matrix[i][this.constants.radius] = -matrixWeight;
            matrix[this.constants.radius][i] = -matrixWeight;
        }
        matrix[this.constants.radius][this.constants.radius] = 1 + ((matrixSize - 1) * 2) * matrixWeight;
        return matrix;
    }
}

export { 
    ImageSharpenFilter
};
