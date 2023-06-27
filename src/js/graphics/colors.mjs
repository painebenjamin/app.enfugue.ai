/** @module graphics/colors */
import { clamp } from "../base/helpers.mjs";

/**
 * This class allows for interpolating between colors.
 */
class ColorScale {
    /**
     * @param array<array<int>> An array of 3-tuple int [r,g,b]
     */
    constructor(colors) {
        this.colors = colors;
        this.length = colors.length;
    }

    /**
     * Gets the color at a certain scale point
     *
     * @param float $t The ratio between 0 and 1
     * @return array<int> The color in RGB
     */
    get(t) {
        t = clamp(t);
        
        if (t === 0) {
            return this.colors[0];
        } else if(t === 1) {
            return this.colors[this.length-1];
        }

        let colorRatio = (this.length - 1) * t,
            colorStart = Math.floor(colorRatio),
            colorEnd = Math.ceil(colorRatio),
            colorMultiplier = colorRatio - colorStart,
            [r0, g0, b0] = this.colors[colorStart],
            [r1, g1, b1] = this.colors[colorEnd],
            [r, g, b] = [
                r0 + ((r1 - r0) * colorMultiplier),
                g0 + ((g1 - g0) * colorMultiplier),
                b0 + ((b1 - b0) * colorMultiplier)
            ];
        
        return [
            clamp(Math.round(r), 0, 255),
            clamp(Math.round(g), 0, 255),
            clamp(Math.round(b), 0, 255)
        ];

    }
}

/**
 * This small helper function gets the contrast color against
 * a certian background color.
 *
 * @param int $r The red value of the color
 * @param int $g The green value of the color
 * @param int $b The blue value of the color
 * @return string The text color that contrasts best
 */
const getTextColorForBackground = (r, g, b) => {
    return r*0.299 + g*0.587 + b*0.114 > 186
        ? "#000000"
        : "#FFFFFF"
};

export { ColorScale, getTextColorForBackground };
