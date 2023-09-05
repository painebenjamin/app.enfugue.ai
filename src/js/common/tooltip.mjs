/** @module common/tooltip */
import { DOMWatcher } from '../base/watcher.mjs';
import { ElementBuilder } from '../base/builder.mjs';

const trackingConfiguration = {
        name: 'Tooltip tracker',
        childList: true,
        subtree: true,
        attributes: true,
        debug: false,
        containerClassName: 'tooltip-container'
    },
    E = new ElementBuilder(),
    offsetConfiguration = {
        x: 10,
        y: 10
    },
    activationDelay = 150,
    checkInterval = 100;

let enableTimer, checkIntervalTimer;

/**
 * This class enables any element to add 'data-tooltip' attributes, and a
 * tooltip will be shown near the cursor when it hovers over that element.
 * The DOMWatcher lets us do this whenever any element is modified, so no need to initialize anything.
 */
class TooltipHelper extends DOMWatcher {
    /**
     * @param object $configuration The tracking configuration overrides. Optional.
     */
    constructor(configuration) {
        if (configuration === undefined) configuration = trackingConfiguration;
        configuration.initialize = false;
        super(configuration);
        this.filterFunction = this.tooltipFilter;
        this.initializeFunction = this.tooltipInitialize;
        this.updateFunction = this.tooltipUpdate;

        this.tooltipContainer = E.div().class(configuration.containerClassName || trackingConfiguration.containerClassName);

        document.body.appendChild(this.tooltipContainer.render());
        this.initialize();
    }

    /**
     * The update function checks if a node's tooltip changed while it's tooltip is shown
     */
    tooltipUpdate(node) {
        if (node == this.tooltipNode) {
            this.tooltipContainer.content(
                node.getAttribute('data-tooltip')
            );
        }
    }


    /**
     * The filter function checks the nodes data-tooltip attribute, if possible.
     * Check if it's an HTMLElement to avoid text nodes.
     * @param node $node an HTMLElement, hopefully
     */
    tooltipFilter(node) {
        return node instanceof HTMLElement && node.hasAttribute('data-tooltip');
    }

    /**
     * Initializes the function.
     * @param node $node An HTMLElement
     */
    tooltipInitialize(node) {
        if (node.hasAttribute('data-tooltip-initialized')) {
            return;
        }

        let self = this,
            positionElement = function (mouseX, mouseY) {
                window.requestAnimationFrame(() => {
                    let tooltip = self.tooltipContainer,
                        width = window.innerWidth,
                        height = window.innerHeight;

                    if (mouseX > width / 2) {
                        tooltip.css('left', null);
                        tooltip.css(
                            'right',
                            width - mouseX + offsetConfiguration.x
                        );
                    } else {
                        tooltip.css('right', null);
                        tooltip.css('left', mouseX + offsetConfiguration.x);
                    }

                    if (mouseY > height / 2) {
                        tooltip.css('top', null);
                        tooltip.css(
                            'bottom',
                            height - mouseY + offsetConfiguration.y
                        );
                    } else {
                        tooltip.css('bottom', null);
                        tooltip.css('top', mouseY + offsetConfiguration.y);
                    }
                });
            },
            activate = function () {
                clearInterval(checkIntervalTimer);
                self.tooltipContainer.addClass('active');
                checkIntervalTimer = setInterval(() => {
                    let tooltipElapsedTime = self.tooltipTime === null
                        ? 0 
                        : ((new Date()).getTime() - self.tooltipTime);
                    if (!self.watchedNode.contains(node) && tooltipElapsedTime > 1000) {
                        deactivate();
                    }
                }, checkInterval);
            },
            deactivate = function () {
                self.tooltipNode = null;
                self.tooltipContainer.removeClass('active');
                clearInterval(checkIntervalTimer);
            },
            mouseMove = function (e) {
                positionElement(e.clientX, e.clientY);
            },
            mouseEnter = function (e) {
                self.tooltipTime = (new Date()).getTime();
                self.tooltipNode = node;
                self.tooltipContainer.content(node.getAttribute('data-tooltip'));
                positionElement(e.clientX, e.clientY);
                enableTimer = setTimeout(activate, activationDelay);
                node.addEventListener('mousemove', mouseMove, true);
            },
            mouseLeave = function (e) {
                clearTimeout(enableTimer);
                deactivate();
                node.removeEventListener('mousemove', mouseMove);
            },
            touchElsewhere = function (e) {
                clearTimeout(enableTimer);
                deactivate();
                window.removeEventListener('touchstart', touchElsewhere);
            },
            touch = function (e) {
                e.stopPropagation();
                e.preventDefault();
                self.tooltipNode = node;
                self.tooltipContainer.content(node.getAttribute('data-tooltip'));
                positionElement(e.touches[0].clientX, e.touches[0].clientY);
                enableTimer = setTimeout(activate, activationDelay);
                window.addEventListener('touchstart', touchElsewhere, true);
            };
        
        node.addEventListener('mouseenter', mouseEnter, true);
        node.addEventListener('mouseleave', mouseLeave), true;
        node.addEventListener('touchstart', touch, true);
        node.setAttribute('data-tooltip-initialized', new Date().getTime());
    }
}

export { TooltipHelper };
