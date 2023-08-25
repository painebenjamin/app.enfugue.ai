/** @module view/base */
import { ElementBuilder } from '../base/builder.mjs';
import { MutexLock } from '../base/mutex.mjs';
import { isEmpty, deepClone } from '../base/helpers.mjs';

const E = new ElementBuilder({
    tabs: 'enfugue-tabs',
    tab: 'enfugue-tab',
    tabContent: 'enfugue-tab-content'
});

/**
 * A View represents a DOMElement with additional state and context.
 */
class View {
    /**
     * @var string The tag name, default div.
     */
    static tagName = 'div';

    /**
     * @var string The class name, default view.
     */
    static className = 'view';

    /**
     * @var string The class to add to the view when it is loading.
     */
    static loaderClassName = 'loader';

    /**
     * @var array<string> An array of additional classes.
     */
    static classList = [];

    constructor(config) {
        this.config = config;
        this.additionalClasses = deepClone(this.constructor.classList);
        this.hidden = false;
        this.lock = new MutexLock();
    }

    /**
     * A callback to perform when resized.
     */
    resize() {}

    /**
     * Hide the element on the page.
     */
    hide() {
        this.hidden = true;
        this.lock.acquire().then((release) => {
            if (this.node !== undefined) {
                this.node.hide();
            }
            release();
        });
        return this;
    }

    /**
     * Hides the elements parent on the page.
     */
    hideParent() {
        this.hidden = true;
        this.lock.acquire().then((release) => {
            if (this.node !== undefined && this.node.element !== undefined) {
                this.previousDisplay = this.node.element.parentElement.style.display;
                this.node.element.parentElement.classList.add("hidden");
            }
            release();
        });
        return this;
    }

    /**
     * Show the element on the page.
     */
    show() {
        this.hidden = false;
        this.lock.acquire().then((release) => {
            if (this.node !== undefined) {
                this.node.show();
            }
            release();
        });
        return this;
    }
    
    /**
     * Shows the elements parent on the page.
     */
    showParent(defaultDisplay = "block") {
        this.hidden = false;
        this.lock.acquire().then((release) => {
            if (this.node !== undefined && this.node.element !== undefined) {
                this.node.element.parentElement.classList.remove("hidden");
            }
            release();
        });
        return this;
    }


    /**
     * Add a class to memory, reflecting in the DOM if possible.
     *
     * @param string $className The class to add.
     */
    addClass(className) {
        this.additionalClasses.push(className);
        this.lock.acquire().then((release) => {
            if (this.node !== undefined) {
                this.node.addClass(className);
            }
            release();
        });
        return this;
    }

    /**
     * Remove a class to memory, reflecting in the DOM if possible.
     *
     * @param string $className The class name to remove.
     */
    removeClass(className) {
        this.additionalClasses = this.additionalClasses.filter(
            (cls) => cls !== className
        );
        this.lock.acquire().then((release) => {
            if (this.node !== undefined) {
                this.node.removeClass(className);
            }
            release();
        });
        return this;
    }

    /**
     * Check for presence of a classname.
     *
     * @param string $className The name to test for.
     * @return bool Whether or not the view has this class.
     */
    hasClass(className) {
        if (this.additionalClasses.indexOf(className) !== -1) {
            return true;
        } else if (this.node !== undefined) {
            return this.node.hasClass(className);
        }
        return false;
    }

    /**
     * Toggle the presence of a classname.
     *
     * @param string $className The class name to toggle
     */
    toggleClass(className) {
        if (this.hasClass(className)) {
            return this.removeClass(className);
        }
        return this.addClass(className);
    }

    /**
     * This creates the node, adding classes and doing what is necessary
     * to show/hide the view. Implementing classes should call `await super.build()`,
     * then fill the returned object with its view content.
     */
    async build() {
        return this.lock.acquire().then((release) => {
            let node = E.createElement(this.constructor.tagName);
            if (!isEmpty(this.constructor.className)) {
                node.addClass(this.constructor.className);
            }

            for (let className of this.additionalClasses) {
                node.addClass(className);
            }

            if (this.hidden) {
                node.hide();
            }
            release();
            return node;
        });
    }

    /**
     * The main function to get the DOMElement, this either returns the in-memory element
     * or builds a new element.
     */
    async getNode() {
        if (this.node === undefined) {
            this.node = await this.build();
        }
        return this.node;
    }

    /**
     * Calls render() on the element.
     *
     * @return DocumentElement The rendered DOM element.
     */
    async render() {
        return (await this.getNode()).render();
    }

    /**
     * Add the 'loader' classes.
     */
    async loading() {
        this.addClass(this.constructor.loaderClassName).addClass('loading').addClass('disabled');
        return this;
    }

    /**
     * Remove the 'loader' classes.
     */
    async doneLoading() {
        this.removeClass('loading').removeClass('disabled');
        return this;
    }
}

/**
 * The ShadowView acts like a view but does not have presence on the DOM
 */
class ShadowView extends View {
    /**
     * @var string Override the tagName to null.
     */
    static tagName = null;

    /**
     * Override build to do nothing.
     */
    async build() {
        return this.lock.acquire().then((release) => {
            let node = E.createShadow();
            release();
            return node;
        });
    }
}

/**
 * The ParentView acts as a collection of views.
 */
class ParentView extends View {
    /**
     * On construction, initialize child array.
     */
    constructor(config) {
        super(config);
        this.children = [];
    }

    /**
     * Gets a child at a specified index
     */
    getChild(index) {
        return this.children[index];
    }

    /**
     * Returns true if this is empty.
     */
    isEmpty() {
        return this.children.length === 0;
    }

    /**
     * Empty the children in memory and on page.
     */
    async empty() {
        this.children = [];
        if (this.node !== undefined) {
            this.node.empty();
        }
        return this;
    }

    /**
     * Add a child in memory and on page at an index.
     * 
     * @param int $index The index to insert at.
     * @param class $childClass The class to instantiate.
     * @param ... Any number of additional arguments to the constructor.
     * @return object The instantiated child class.
     */
    async insertChild(index, childClass) {
        let child;
        if (typeof childClass == 'function') {
            child = new childClass(
                this.config,
                ...Array.from(arguments).slice(2)
            );
        } else if (childClass instanceof View) {
            child = childClass;
        } else {
            console.trace();
            console.error(childClass);
            throw `Cannot add child of type ${typeof childClass}`;
        }
        child.parent = this;
        this.children = this.children
            .slice(0, index)
            .concat([child])
            .concat(this.children.slice(index));
        if (this.node !== undefined) {
            this.node.insert(index, await child.getNode());
        }
        return child;
    }

    /**
     * Add a child to the end of the child list.
     *
     * @param class $childClass The class to instantiate.
     * @param ... Any number of additional arguments to the constructor.
     * @return object The instantiated child class.
     */
    async addChild(childClass) {
        return this.insertChild(
            this.children.length,
            childClass,
            ...Array.from(arguments).slice(1)
        );
    }

    /**
     * Removes a child from the parent by memory location.
     *
     * @param object $removedChild The child to remove.
     */
    removeChild(removedChild) {
        for (let child of this.children) {
            if (child == removedChild) {
                this.children = this.children.filter((c) => c != removedChild);
                if (this.node !== undefined) {
                    this.node.remove(removedChild.node);
                }
                return;
            }
        }
        throw 'Cannot find child to remove.';
    }

    /**
     * Build the parent node and add child nodes.
     */
    async build() {
        let node = await super.build();
        for (let child of this.children) {
            node.append(await child.getNode());
        }
        return node;
    }
}

/**
 * The TabbedView allows selecting between a number of different child views
 */
class TabbedView extends View {
    /**
     * @var string The custom tag name
     */
    static tagName = 'enfugue-tabbed-view';

    /**
     * On construction, create tab map.
     */
    constructor(config) {
        super(config);
        this.tabs = {};
        this.activeTab = null;
    }

    /**
     * Adds a new tab.
     *
     * @param string $tabTitle The text of the tab to show on the tab selector.
     * @param mixed $tabContent The content of the tab.
     */
    async addTab(tabTitle, tabContent) {
        this.tabs[tabTitle] = tabContent;
    }

    /**
     * Activates a tab.
     *
     * @param string $tabTitle The tab to activate.
     */
    async activateTab(tabTitle) {
        this.activeTab = tabTitle;
        if (this.node !== undefined) {
            let tabs = this.node.findAll(E.getCustomTag('tab'));
            for (let tab of tabs) {
                if (tab.getText() == tabTitle) {
                    tab.addClass('active');
                } else {
                    tab.removeClass('active');
                }
            }
            let content = this.tabs[tabTitle];
            if (content instanceof View) {
                content = await content.getNode();
            }
            this.node.find(E.getCustomTag('tabContent')).content(content);
        }
        return this;
    }

    /**
     * When building, add tab headers and first rendered tab.
     */
    async build() {
        let node = await super.build(),
            tabs = E.tabs(),
            tabContent = E.tabContent();

        for (let titleName in this.tabs) {
            let tab = E.tab()
                .content(titleName)
                .on('click', () => this.activateTab(titleName));
            tabs.append(tab);
            if (this.activeTab === null) {
                this.activeTab = titleName;
            }
            if (this.activeTab === titleName) {
                tab.addClass('active');
                let content = this.tabs[titleName];
                if (content instanceof View) {
                    content = await content.getNode();
                }
                tabContent.content(content);
            }
        }
        return node.content(tabs, tabContent);
    }
}

export { View, ParentView, TabbedView, ShadowView };
