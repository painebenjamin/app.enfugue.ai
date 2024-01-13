/** @module view/controls.mjs */
import { View } from './base.mjs';
import { ElementBuilder } from '../base/builder.mjs';
import { isEmpty } from '../base/helpers.mjs';

const E = new ElementBuilder();

class ControlsHelperView extends View {
    /**
     * @var string CSS classname
     */
    static className = "controls-helper-view";

    /**
     * @var string Text to show on screen
     */
    static helpLabel = "Canvas Controls";

    /**
     * @var string Text to show on mouseover
     */
    static helpText = "<strong>General</strong><ul style='list-style-type: bullet; padding: 1em;'><li>Move the entire canvas (pan) by placing your cursor over it then holding down <code>Middle-Mouse-Button</code>, or alternatively <code>Ctrl+Left-Mouse-Button</code> or <code>Alt+Left-Mouse-Button</code> (<code>Option⌥+Left-Mouse-Button</code> on MacOS), and move the canvas around.</li><li>Zoom in and out using the scroll wheel or scroll gestures. You can also click the + and - icons in the bottom-right-hand corner. Click 'RESET' at any time to bring the canvas back to the initial position.</li></ul><strong>Painting</strong><ul style='list-style-type: bullet; padding: 1em;'><li>Use the scroll wheel or scroll gestures to increase or descrease the size of your brush.</li><li>Hold <code>Ctrl</code> when scrolling up/down to stop this behavior and instead perform the general behavior of zooming in/out.</li><li>Use <code>Left-Mouse-Button</code> to draw, or <code>Alt+Left-Mouse-Button</code> to erase.</li><li>After painting and releasing <code>Left-Mouse-Button</code>, hold <code>shift</code> when you begin painting again to draw a straight line between the previous final and your current position using the current brush.</li></ul><strong>Motion Vectors</strong><ul style='list-style-type: bullet; padding: 1em 1em 0 1em;'><li>Click the <code>Left-Mouse-Button</code> on an empty portion of the canvas to start selecting existing points with a rectangular selector. Hold <code>shift</code> while doing this to add the selected points to your current selection, instead of replacing it. When you left-click on a point on the canvas, it will be grabbed and moved, optionally with shift as well to move all points at once. Left-clicking on a spline instead will select and move the entire spline.</li><li>Click <code>Alt+Left-Mouse-Button</code> on an empty section of the canvas to draw a new linear motion vector. Move your mouse and release it to draw a line between those points. When you use <code>Alt</code> and left-click on a point instead of the canvas, the point will be deleted. Alt-left-clicking a spline will add a new point along the segment your mouse is over.</li><li>Hold <code>Ctrl+Shift</code> and click the left-mouse-button anywhere on the canvas to rotate all selected points about their center.</li><li>When points are selected, press <code>Delete</code> on your keyboard to delete them.</li><li>When points are selected, press <code>Ctrl+C</code> on your keyboard to copy them. You must select at least two points on a spline for it to be copied.</li><li>Holding <code>Ctrl</code> while left-clicking will resume the previous behavior of moving the canvas.</ul>";

    /**
     * On build, add helper texts
     */
    async build() {
        let node = await super.build();
        node.content(this.constructor.helpLabel)
            .data(
                "tooltip",
                this.constructor.helpText
            );
        return node;
    }
}

/**
 * Shows the control text in a node instead
 */
class ControlsView extends View {
    /**
     * @var string classname
     */
    static className = "controls-view";

    /**
     * On build, assemble help paragraph
     */
    async build() {
        let node = await super.build();
        node.content(ControlsHelperView.helpText);
        return node;
    }
}
export { ControlsHelperView, ControlsView };
