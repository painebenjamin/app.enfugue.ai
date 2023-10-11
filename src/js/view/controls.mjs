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
    static helpText = "<strong>General</strong><ul style='list-style-type: bullet; padding: 1em;'><li>Move the entire canvas (pan) by placing your cursor over it then holding down <code>Middle-Mouse-Button</code>, or alternatively <code>Ctrl+Left-Mouse-Button</code> or <code>Alt+Left-Mouse-Button</code> (<code>Option⌥+Left-Mouse-Button</code> on MacOS), and move the canvas around.</li><li>Zoom in and out using the scroll wheel or scroll gestures. You can also click the + and - icons in the bottom-right-hand corner. Click 'RESET' at any time to bring the canvas back to the initial position.</li></ul><strong>Painting</strong><ul style='list-style-type: bullet; padding: 1em 1em 0 1em;'><li>Use the scroll wheel or scroll gestures to increase or descrease the size of your brush.</li><li>Hold <code>Ctrl</code> when scrolling up/down to stop this behavior and instead perform the general behavior of zooming in/out.</li><li>Use <code>Left-Mouse-Button</code> to draw, or <code>Alt+Left-Mouse-Button</code> to erase.</li><li>After painting and releasing <code>Left-Mouse-Button</code>, hold <code>shift</code> when you begin painting again to draw a straight line between the previous final and your current position using the current brush.</li></ul>";

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

export { ControlsHelperView };
