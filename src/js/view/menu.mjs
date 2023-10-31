/** @module view/menu */
import { View, ParentView } from "./base.mjs";
import { ElementBuilder } from "../base/builder.mjs";
import { isEmpty } from "../base/helpers.mjs";

const E = new ElementBuilder({
    categoryHeader: "enfugue-menu-category-header",
    categoryItems: "enfugue-menu-category-items",
    categoryButton: "enfugue-menu-category-button",
    toolbarOperations: "enfugue-toolbar-operations",
    toolbarInputs: "enfugue-toolbar-inputs",
    toolbarItem: "enfugue-toolbar-item"
});

/**
 * Formats a name with a shortcut
 */
function formatName(name, shortcut = null) {
    if (isEmpty(shortcut)) {
        return name;
    }
    let formatted = "<span>",
        highlighted = false;

    for (let i = 0; i < name.length; i++) {
        let character = name[i];
        if (!highlighted && character.toLowerCase() === shortcut.toLowerCase()) {
            formatted += `</span><span class="shortcut">${character}</span>`;
            highlighted = true;
        } else {
            formatted += character;
        }
    }
    if (formatted[formatted.length-1] !== ">") {
        formatted += "</span>";
    }
    return formatted;
}

/**
 * The MenuView allows Categories and items
 */
class MenuView extends ParentView {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-menu";

    /**
     * Turns off all categories
     */
    hideCategories() {
        for (let child of this.children) {
            if (child instanceof MenuCategoryView){
                child.removeClass("active");
            }
        }
    }

    /**
     * Shows a category by shortcut
     */
    async fireCategoryShortcut(key) {
        for (let child of this.children) {
            if (child instanceof MenuCategoryView) {
                if (!isEmpty(child.shortcut) && child.shortcut.toLowerCase() === key.toLowerCase()) {
                    this.toggleCategory(child.name);
                    return;
                }
                if (child.hasClass("active")) {
                    for (let grandchild of child.children) {
                        if (!isEmpty(grandchild.shortcut) && grandchild.shortcut.toLowerCase() === key.toLowerCase()) {
                            await grandchild.activate();
                            this.hideCategories();
                            return;
                        }
                    }
                }
            }
        }
    }

    /**
     * Starts a hide timer to hide self
     */
    startHideTimer() {
        this.hideTimer = setTimeout(() => {
            this.hideCategories();
        }, 500);
    }

    /**
     * Toggles a specific category
     *
     * @param string $name The name of the category to hide
     * @return bool True if it was toggled on, false if not
     */
    toggleCategory(name) {
        let found = false, newValue;
        clearTimeout(this.hideTimer);
        for (let child of this.children) {
            if (child instanceof MenuCategoryView){
                if (child.name === name) {
                    found = true;
                    newValue = !child.hasClass("active");
                    child.toggleClass("active");
                } else {
                    child.removeClass("active");
                }
            }
        }
        if (!found) {
            console.warn("No category named", name);
        }
        return newValue;
    }

    /**
     * Removes a specific category
     *
     * @param string $name The name of the category
     * @return bool True if removed, false if not
     */
    removeCategory(name) {
        for (let child of this.children) {
            if (child instanceof MenuCategoryView && child.name === name){
                this.removeChild(child);
                return true;
            }
        }
        return false;
    }
    
    /**
     * Adds a new category
     *
     * @param string $name The name of the category
     * @return MenuCategoryView The instantiated category
     */
    addCategory(name, shortcut) {
        let index = 0;
        while (this.children[index] instanceof MenuCategoryView) index++;
        return this.insertChild(index, MenuCategoryView, name, shortcut);
    }

    /**
     * When building, bind the window to hide when clicking elsewhere
     */
    async build() {
        let node = await super.build();
        node.on("click", (e) => e.stopPropagation());
        window.addEventListener("click", () => this.hideCategories());
        return node;
    }
}

/**
 * The MenuCategoryView is added to the MenuView
 */
class MenuCategoryView extends ParentView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-menu-category";

    /**
     * @var object $config The base configuration object
     * @var string $name The name of the category
     */
    constructor(config, name, shortcut) {
        super(config);
        this.name = name;
        this.shortcut = shortcut;
        this.buttons = [];
    }

    /**
     * Sets the name after construction
     *
     * @param string $name The new name
     */
    setName(name) {
        this.name = name;
        if (this.node !== undefined) {
            this.node
                .find(E.getCustomTag("categoryHeader"))
                .find("span")
                .content(formatName(this.name, this.shortcut));
        }
    }

    /**
     * Adds a button to this category view.
     *
     * @param string $icon The icon classes 
     * @param callable $callback The callback to fire when the button is clicked
     * @return DOMElement The button element.
     */
    addButton(icon, callback) {
        let newButton = E.categoryButton()
            .content(E.i().class(icon))
            .on("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                callback();
            });
        this.buttons.push(newButton);
        if (this.node !== undefined) {
            this.node.find(E.getCustomTag("categoryHeader")).append(newButton);
        }
        return newButton;
    }

    /**
     * Inserts a child item at a specific index.
     *
     * @param int $index The index to insert
     * @param class $childClass The child class to instantiate.
     * @param ... Any number of additional arguments are passed to the constructor.
     * @return An instance of $childClass
     */
    async insertChild(index, childClass) {
        let child = new childClass(
            this.config,
            ...Array.from(arguments).slice(2)
        );
        this.children = this.children
            .slice(0, index)
            .concat([child])
            .concat(this.children.slice(index));
        if (this.node !== undefined) {
            this.node.insert(index + 1, await child.getNode());
        }
        return child;
    }

    /**
     * Adds a sub-category view.
     *
     * @param string $name The name of the category
     */
    async addCategory(name) {
        let index = 0;
        while (this.children[index] instanceof MenuCategoryView) index++;
        return this.insertChild(index, MenuCategoryView, name);
    }

    /**
     * Adds an icon item view.
     *
     * @param string $name The text to display.
     * @param string $icon The icon classes to display
     * @return IconItemView
     */
    async addItem(name, icon, shortcut) {
        let itemView = await this.addChild(IconItemView, name, icon, shortcut);
        itemView.onClick(() => { this.removeClass("active"); });
        return itemView;
    }

    /**
     * Build the header and sub-items
     */
    async build() {
        let node = await super.build(),
            header = E.categoryHeader().content(E.span().content(formatName(this.name, this.shortcut)));

        for (let button of this.buttons) {
            header.append(button);
        }

        node.prepend(header)
            .on("mouseenter", () => this.parent.toggleCategory(this.name))
            .on("mouseleave", () => this.parent.startHideTimer());

        return node;
    }
}

/**
 * The MenuItemView is an individual item in a MenuCategoryView
 */
class MenuItemView extends View {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-menu-item";

    /**
     * @param object $config The configuration object
     * @param string $name The name of the menu item (text)
     */
    constructor(config, name, shortcut) {
        super(config);
        this.name = name;
        this.shortcut = shortcut;
        this.callbacks = [];
    }

    /**
     * Adds a callback that is fired when the item is clicked.
     *
     * @param callable $callback The callback.
     */
    onClick(callback) {
        this.callbacks.push(callback);
    }

    /**
     * Sets the menu item name after initialization.
     *
     * @param string $name The new name.
     */
    setName(name) {
        this.name = name;
        if (this.node !== undefined) {
            this.node.find("span").content(this.name);
        }
    }

    /**
     * Fires callbacks
     */
    async activate() {
        for (let callback of this.callbacks) {
            await callback();
        }
    }

    /**
     * Builds the node and binds events.
     */
    async build() {
        let node = await super.build();
        node.prepend(E.span().content(formatName(this.name, this.shortcut)));
        node.on("click", async(e) => {
            e.preventDefault();
            this.activate()
        });
        return node;
    }
}

/**
 * The IconItemView is like a MenuItemView but includes an icon.
 */
class IconItemView extends MenuItemView {
    /**
     * @param object $config The base config object.
     * @param string $name The name (text) of the item.
     * @param string $icon The icon classes or image source
     */
    constructor(config, name, icon, shortcut) {
        super(config, name, shortcut);
        this.icon = icon;
    }

    /**
     * Sets the icon after initialization
     */
    setIcon(newIcon) {
        this.icon = newIcon;
        if (!isEmpty(this.node)) {
            if (newIcon.startsWith("http")) {
                this.node.find("img").src(newIcon);
            } else {
                this.node.find("i").class(newIcon)
            }
        }
    }

    /**
     * On build, append icon.
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.icon)) {
            if (this.icon.startsWith("http") || this.icon.startsWith("/")) {
                node.prepend(E.img().src(this.icon));
            } else {
                node.prepend(E.i().class(this.icon));
            }
        }
        return node;
    }
}

/**
 * The sidebar simply adds sidebar icon items
 */
class SidebarView extends ParentView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-sidebar";

    /**
     * Override this to use SidebarItemView
     */
    addItem(name, icon) {
        return this.addChild(SidebarItemView, name, icon);
    }
}

/**
 * This class simply changes the tag name for CSS purposes
 */
class SidebarItemView extends IconItemView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-sidebar-item";
}

/**
 * The toolbar class allows for some inputs and items.
 */
class ToolbarView extends ParentView {
    /**
     * @var string The custom tag name
     */
    static tagName = "enfugue-toolbar";
    
    /**
     * Override this to use ToolbbarItemView
     */
    addItem(name, icon) {
        return this.addChild(ToolbarItemView, name, icon);
    }
}

/**
 * The Toolbar is a smaller version of the menu.
 */
class ToolbarItemView extends IconItemView {
    /**
     * @var string The custom tag name.
     */
    static tagName = "enfugue-toolbar-item";
    
    /**
     * On build, set tooltip.
     */
    async build() {
        let node = await super.build();
        node.data("tooltip", this.name);
        return node;
    }
}

export {
    MenuView,
    MenuItemView,
    MenuCategoryView,
    IconItemView,
    SidebarView,
    SidebarItemView,
    ToolbarView,
    ToolbarItemView
};
