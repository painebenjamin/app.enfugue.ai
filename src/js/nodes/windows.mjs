/** @module nodes/windows */
import { isEmpty } from '../base/helpers.mjs';
import { NodeView } from './base.mjs';
import { NodeEditorView } from './editor.mjs';
import { ElementBuilder } from '../base/builder.mjs';

const E = new ElementBuilder({
    windowsToolbar: 'enfugue-windows-toolbar',
    toolbarItem: 'enfugue-windows-toolbar-item',
    itemName: 'enfugue-windows-toolbar-item-name',
    itemButton: 'enfugue-windows-toolbar-item-button'
});

/**
 * The WindowView allows for moving around regular HTML contents on the page./
 */
class WindowView extends NodeView {
    /**
     * @var int The snap size of the window - one pixel means no snap.
     */
    static snapSize = 1;

    /**
     * @var bool Disable copying.
     */
    static canCopy = false;

    /**
     * @var object The minimize and maximize buttons.
     */
    static nodeButtons = {
        minimize: {
            icon: 'fas fa-window-minimize',
            callback: function () {
                this.editor.minimizeNode(this);
            }
        },
        maximize: {
            icon: 'fas fa-window-maximize',
            callback: function () {
                this.editor.toggleMaximizeNode(this);
            }
        }
    }

    /**
     * Focus() will also un minimize
     */
    focus() {
        super.focus();
        this.editor.unMinimizeNode(this);
    }
    
    /**
     * Minimize this node.
     */
    minimize() {
        this.editor.minimizeNode(this);
    }
    
    /**
     * Maximize this node.
     */
    maximize() {
        this.editor.toggleMaximizeNode(this);
    }
}

/**
 * This view manages multiple windows at once.
 */
class WindowsView extends NodeEditorView {
    /**
     * @var bool Disable the background of the node editor, only the nodes receives events
     */
    static disableCursor = true;

    /**
     * @var int|str Enable auto-width
     */
    static canvasWidth = 'auto';
    
    /**
     * @var int|str Enable auto-height
     */
    static canvasHeight = 'auto';

    /**
     * @var str The class name allows for additional CSS styles.
     */
    static className = 'windows';

    constructor(config) {
        super(config);
        this.minimized = [];
    }

    /**
     * Minimize a window.
     *
     * @param NodeView $node The node to minimize.
     */
    minimizeNode(node) {
        node.addClass("minimized");
        this.minimized.push(node.addClass('minimized'));
        let expectedLeftPosition = 0;
        if (this.node !== undefined) {
            let toolbar = this.node.find(E.getCustomTag('windowsToolbar'));
            let otherToolbarItems = toolbar.findAll(E.getCustomTag('toolbarItem'));
            let lastToolbarItem = otherToolbarItems.slice(-1).pop();
            
            if (!isEmpty(lastToolbarItem)) {
                expectedLeftPosition = lastToolbarItem.element.offsetLeft;
            }

            let toolbarItem = E.toolbarItem()
                .data("node-id", node.node.elementId)
                .content(
                    E.itemName().content(node.name),
                    E.itemButton()
                        .content(E.i().class('fas fa-window-close'))
                        .on('click', (e) => {
                            this.removeNode(node);
                            node.closed();
                        })
                )
                .on('click', (e) => {
                    this.unMinimizeNode(node);
                });
            this.node
                .find(E.getCustomTag('windowsToolbar'))
                .append(toolbarItem);
        }
        node.setDimension(expectedLeftPosition, this.height, 0, 0, false);
    }
    
    /**
     * Un-minimize a window
     *
     * @param NodeView $node The node to un-minimize.
     */
    unMinimizeNode(node) {
        this.minimized = this.minimized.filter((m) => m !== node);
        node.resetDimension();
        let toolbar = this.node.find(E.getCustomTag("windowsToolbar")),
            toolbarItem = toolbar.find(`[data-node-id='${node.node.elementId}`);
        if (!isEmpty(toolbarItem)) {
            toolbar.remove(toolbarItem);
        }
        setTimeout(() => node.removeClass('minimized'), 250);
    }

    /**
     * Maximizes or un-maximizes a node.
     *
     * @param NodeView $node The node on which to toggle maximization.
     */
    toggleMaximizeNode(node) {
        if (node.maximized == true) {
            node.setDimension(
                node.lastLeft,
                node.lastTop,
                node.lastWidth,
                node.lastHeight
            );
            node.maximized = false;
        } else {
            node.lastLeft = node.left;
            node.lastTop = node.top;
            node.lastWidth = node.width;
            node.lastHeight = node.height;
            node.maximized = true;

            node.setDimension(
                -node.constructor.padding,
                -node.constructor.padding,
                this.width + node.constructor.padding * 2,
                this.height + node.constructor.padding * 2
            );
            node.addClass('focused');
        }
        node.resized();
    }

    /**
     * This is a shorthand helper for spawning a window on the desktop.
     *
     * @param string $name The name of the window.
     * @param View|DOMElement $content The content of the window.
     * @param int $width The width of the window. If not passed, will fill the canvas.
     * @param int $height The height of the window. If not passed, will fill the canvas.
     * @param int $left The number of pixels from the left to spawn. If not passed, will try and spawn a window offset other windows.
     * @param int $top The number of pixels from the top to spawn. If not passed, will try and spawn a window offset other windows.
     * @return WindowView The added node.
     */
    async spawnWindow(name, content, width, height, left, top) {
        if (isEmpty(width)) {
            width = this.width - WindowView.padding * 2;
        }

        if (isEmpty(height)) {
            height = this.height - WindowView.padding * 2;
        }

        if (left === undefined) {
            left = WindowView.padding + document.scrollingElement.scrollLeft;
        }

        if (top === undefined) {
            top = WindowView.padding + document.scrollingElement.scrollTop;
        }

        for (let node of this.nodes) {
            if (
                node.left == left - WindowView.padding &&
                node.top == top - WindowView.padding
            ) {
                left += WindowView.padding;
                top += WindowView.padding;
            }
        }

        let newWindow = await this.addNode(
            WindowView,
            name,
            content,
            left,
            top,
            width,
            height
        );
        newWindow.focus();
        return newWindow;
    }

    /**
     * Triggers all resize callbacks.
     * Also for this class, this will resize windows to fit within the new browser size.
     */
    async windowResized(e) {
        await super.windowResized(e);
        if (this.node !== undefined) {
            let windowWidth = this.node.element.clientWidth,
                windowHeight = this.node.element.clientHeight,
                saveChanges = true;

            for (let node of this.nodes) {
                let overHangLeft = node.left + node.width - windowWidth,
                    overHangBottom = node.top + node.height - windowHeight;

                if (overHangLeft > 0) {
                    if (overHangBottom > 0) {
                        node.setDimension(
                            node.left,
                            node.top,
                            node.width - overHangLeft,
                            node.height - overHangBottom,
                            saveChanges
                        );
                    } else {
                        node.setDimension(
                            node.left,
                            node.top,
                            node.width - overHangLeft,
                            node.height,
                            saveChanges
                        );
                    }
                } else if (overHangBottom > 0) {
                    node.setDimension(
                        node.left,
                        node.top,
                        node.width,
                        node.height - overHangBottom,
                        saveChanges
                    );
                }
            }
        }
    }

    /**
     * Override build() to additionally add the toolbar.
     */
    async build() {
        let node = await super.build(),
            toolbar = E.windowsToolbar();

        for (let minimizedWindow of this.minimized) {
            let toolbarItem = E.toolbarItem()
                .content(
                    E.itemName().content(minimizedWindow.name),
                    E.itemButton()
                        .content(E.i().class('fas fa-window-close'))
                        .on('click', (e) => {
                            this.removeNode(minimizedWindow);
                            minimizedWindow.closed();
                        })
                )
                .on('click', (e) => {
                    this.unMinimizeNode(minimizedWindow);
                    toolbar.remove(toolbarItem);
                });

            toolbar.append(toolbarItem);
        }

        return node.append(toolbar);
    }
}

export { WindowsView, WindowView };
