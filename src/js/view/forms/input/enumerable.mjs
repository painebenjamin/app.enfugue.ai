/** @module view/forms/input/enumerable */
import { isEmpty, jaroWinkler, strip } from "../../../base/helpers.mjs";
import { ElementBuilder } from "../../../base/builder.mjs";
import { InputView } from "./base.mjs";
import { StringInputView } from "./string.mjs";

const E = new ElementBuilder({
    searchList: "enfugue-search-list",
    searchListItem: "enfugue-search-list-item",
    searchSelection: "enfugue-search-list-selection",
    repeatableInput: "enfugue-repeatable-input",
    repeatableItem: "enfugue-repeatable-input-item"
});

/**
 * A superclass for inputs that selected one or more items out of
 * a list of items - selects, picking one out of a few, etc.
 */
class EnumerableInputView extends InputView {
    /**
     * @var object The default option values - can be object or array
     */
    static defaultOptions = {};

    /**
     * Allow passing in options, sort and filter to field conf
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.options = this.fieldConfig.options || {};
        this.sortFunction = this.fieldConfig.sortFunction || ((a, b) => a - b);
        this.filterFunction = this.fieldConfig.filterFunction || ((a, b) => false);
    }

    /**
     * Sets the options to a new set.
     *
     * @param array|object|callable $newOptions The new options for this input view
     */
    setOptions(newOptions) {
        if (Array.isArray(newOptions)) {
            this.options = newOptions.reduce((carry, item) => {
                carry[item] = item;
                return carry;
            }, {});
        } else {
            this.options = newOptions;
        }
        if (this.node !== undefined) {
            this.rebuildOptions();
        }
    }

    /**
     * Sorts the options.
     *
     * @param callable $sortFunction The sort compare function
     */
    async sortOptions(sortFunction) {
        this.sortFunction = sortFunction;
        if (this.node !== undefined) {
            await this.rebuildOptions();
        }
    }

    /**
     * Filters the options.
     *
     * @param callable $filterFunction The function to return t/f on each option.
     */
    async filterOptions(filterFunction) {
        this.filterFunction = filterFunction;
        if (this.node !== undefined) {
            await this.rebuildOptions();
        }
    }

    /**
     * Sorts and filters at once
     *
     * @param callable $sortFunction The sort compare function
     * @param callable $filterFunction The function to return t/f on each option.
     * */
    async sortFilterOptions(sortFunction, filterFunction) {
        this.sortFunction = sortFunction;
        this.filterFunction = filterFunction;
        if (this.node !== undefined) {
            await this.rebuildOptions();
        }
    }

    /**
     * This gets the options as an object, no matter the configured type.
     *
     * @return object The key-value map of option names and labels
     */
    async getOptions() {
        let options = this.options,
            defaultOptions = this.constructor.defaultOptions;

        if (typeof options === "function") {
            options = await this.options();
        }

        if (Array.isArray(options)) {
            options = options.reduce((carry, item) => {
                carry[item] = item;
                return carry;
            }, {});
        }

        if (typeof defaultOptions === "function") {
            defaultOptions = await defaultOptions();
        }

        if (Array.isArray(defaultOptions)) {
            defaultOptions = defaultOptions.reduce((carry, item) => {
                carry[item] = item;
                return carry;
            }, {});
        }

        let allOptions = { ...defaultOptions, ...options },
            allOptionKeys = Object.getOwnPropertyNames(allOptions),
            visibleOptions = {};

        allOptionKeys.sort((key1, key2) =>
            this.sortFunction(key1, key2, allOptions[key1], allOptions[key2])
        );
        for (let optionName of allOptionKeys) {
            if (!this.filterFunction(optionName, allOptions[optionName])) {
                visibleOptions[optionName] = allOptions[optionName];
            }
        }
        return visibleOptions;
    }

    /**
     * Builds the options node in the parent node.
     * Called at build time or when options are changed.
     *
     * @param DOMElement $node The parent node
     * @param object $options The options.
     */
    buildOptions(node, options) {
        return;
    }

    /**
     * Rebuilds the options array.
     */
    async rebuildOptions() {
        if (this.node !== undefined) {
            this.buildOptions(this.node, await this.getOptions());
        }
    }

    /**
     * Builds the node, placing options in the parent node.
     */
    async build() {
        let node = await super.build();
        this.buildOptions(node, await this.getOptions());
        return node;
    }
}

/**
 * This is the default enumerable implementation, a select dropdown.
 */
class SelectInputView extends EnumerableInputView {
    /**
     * @var string The tag name
     */
    static tagName = "select";

    /**
     * @var bool Whether or not to always show 'empty' option.
     */
    static alwaysShowEmpty = false;

    /**
     * @var bool Whether or not the user can select no options.
     */
    static allowEmpty = false;

    /**
     * Builds the <option> elements.
     */
    async buildOptions(node, options) {
        let placeholder = this.placeholder;

        if (isEmpty(placeholder)) {
            placeholder = 'Select One';
        }

        let nodeContent = [];

        if (
            isEmpty(this.value) ||
            this.constructor.alwaysShowEmpty ||
            this.constructor.allowEmpty
        ) {
            nodeContent.push(
                E.option()
                    .selected(true)
                    .disabled(!this.constructor.allowEmpty)
                    .value('')
                    .content(placeholder)
            );
        }

        for (let optionName in options) {
            let optionNode = E.option()
                .value(optionName)
                .content(options[optionName]);
            if (this.value == optionName) {
                optionNode.selected(true);
            }
            nodeContent.push(optionNode);
        }
        return node.content(...nodeContent);
    }

    /**
     * Sets the value, selecting the option child node.
     */
    setValue(newValue, triggerChange) {
        if (this.node !== undefined) {
            this.node.value(newValue);
            for (let optionNode of this.node.findAll("option")) {
                if (optionNode.value() == this.value) {
                    optionNode.selected(true);
                } else {
                    optionNode.selected(false);
                }
            }
        }
        super.setValue(newValue, triggerChange);
    }

    /**
     * Builds the node and sets the value if necessary.
     */
    async build() {
        let node = await super.build();
        if (!isEmpty(this.value)) {
            node.value(this.value);
        }
        return node;
    }
}

/**
 * Select one item from a list
 */
class ListInputView extends EnumerableInputView {
    /**
     * @var string The tag name
     */
    static tagName = "ul";

    /**
     * @var string The class name
     */
    static className = "list-input-view";

    /**
     * @var int The maximum number of options to show.
     */
    static maximumOptions = Infinity;

    /**
     * Override getValue to always return the value in memory
     */
    getValue() {
        return this.value;
    }

    /**
     * Override setValue to change the class of the child element
     */
    setValue(newValue, triggerChange) {
        super.setValue(newValue, triggerChange);
        if (this.node !== undefined) {
            for (let childNode of this.node.findAll("li")) {
                if (childNode.data("value") === this.value) {
                    childNode.addClass("selected");
                } else {
                    childNode.removeClass("selected");
                }
            }
        }
    }

    /**
     * When options are built, lay out list items and bind clicks
     */
    buildOptions(node, options) {
        let optionsShown = 0,
            nodeOptions = [];
        for (let optionValue in options) {
            let optionContent = E.span().content(options[optionValue]),
                optionNode = E.li()
                    .content(optionContent)
                    .data("value", optionValue);

            optionNode.on("click", (e) => {
                e.stopPropagation();
                e.preventDefault();
                for (let child of node.children()) {
                    child.removeClass("selected");
                }
                optionNode.addClass("selected");
                this.value = optionValue;
                this.changed();
            });

            if (this.value === optionValue) {
                optionNode.addClass("selected");
            }

            nodeOptions.push(optionNode);
            optionsShown++;
            if (optionsShown >= this.constructor.maximumOptions) {
                break;
            }
        }
        return node.content(...nodeOptions);
    }
}

/**
 * The same as the ListInputView, but lets multiple selections.
 */
class ListMultiInputView extends ListInputView {
    /**
     * @var int The maximum number of items one can select. Default unbounded.
     */
    static maximumSelected = Infinity;

    /**
     * @var string Class names
     */
    static className = "list-input-view multi-list-input-view";

    /**
     * We use empty array as the default value instead of null
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        if (isEmpty(this.value)) {
            this.value = [];
        }
    }

    /**
     * When setting value, select all items and make sure its an array
     */
    setValue(newValue, triggerChange) {
        super.setValue(newValue, triggerChange);
        if (isEmpty(this.value)) {
            this.value = [];
        } else if (!Array.isArray(this.value)) {
            this.value = [this.value];
        }
        if (this.node !== undefined) {
            for (let childNode of this.node.findAll("li")) {
                if (this.value.indexOf(childNode.data("value")) !== -1) {
                    childNode.addClass("selected");
                } else {
                    childNode.removeClass("selected");
                }
            }
        }
    }

    /**
     * Builds options and bind clicks
     */
    buildOptions(node, options) {
        let optionsShown = 0,
            nodeOptions = [];

        for (let optionValue in options) {
            let optionNode = E.li()
                .content(E.span().content(options[optionValue]))
                .data("value", optionValue);

            optionNode.on("click", (e) => {
                e.stopPropagation();
                e.preventDefault();

                if (optionNode.hasClass("selected")) {
                    this.value = this.value.filter((value) => value !== optionValue);
                    optionNode.removeClass("selected");
                } else {
                    if (isEmpty(this.value)) {
                        this.value = [];
                    }
                    this.value.push(optionValue);
                    optionNode.addClass("selected");
                }
                this.changed();
            });

            if (isEmpty(this.value) && this.value.indexOf(optionValue) !== -1) {
                optionNode.addClass("selected");
            }

            nodeOptions.push(optionNode);
            optionsShown++;
            if (optionsShown >= this.constructor.maximumOptions) {
                break;
            }
        }
        return node.content(...nodeOptions);
    }
}

/**
 * This is like the ListInputView but restricts maximum options,
 * this class is used by the SearchListInputView
 */
class SearchListInputListView extends ListInputView {
    /**
     * @var int Keep the maximum options low for display
     */
    static maximumOptions = 100;

    /**
     * @var array<string> Extra classes for this view
     */
    static classList = ["enfugue-search-list-items"];

    /**
     * When built, disable mousewheel movement
     */
    async build() {
        let node = await super.build();
        node.on("mousewheel", (e) => {
            e.stopPropagation();
        });
        return node;
    }
}

/**
 * This extends the ListMultiInputView but restricts maximum options,
 * this class is used by the SearchListInputView
 */
class SearchListMultiInputListView extends ListMultiInputView {
    /**
     * @var int Keep the maximum options low for display
     */
    static maximumOptions = 100;
    
    /**
     * @var array<string> Extra classes for this view
     */
    static classList = [
        "enfugue-search-list-items",
        "enfugue-multi-search-list-items"
    ];

    /**
     * When built, disable mousewheel movement
     */
    async build() {
        let node = await super.build();
        node.on("mousewheel,mouseup", (e) => {
            e.stopPropagation();
        });
        return node;
    }
}

/**
 * This shows a text input and allows a user to select from the closest results.
 * Closeness is measured using jaro-winkler.
 */
class SearchListInputView extends EnumerableInputView {
    /**
     * @var string Custom tag name
     */
    static tagName = "enfugue-search-list";

    /**
     * @var string The placeholder to show it is a search box
     */
    static placeholder = "Start typing to search…";

    /**
     * @var int How many milliseconds after a user finishes input will the search fire
     */
    static searchTimeout = 250;

    /**
     * @var float The closeness threshold below which results will not be shown
     */
    static filterThreshold = 0.9;

    /**
     * @var class The class of the list input that will be shown
     */
    static listInputClass = SearchListInputListView;

    /**
     * @var class The class of the text input that will be shown
     */
    static stringInputClass = StringInputView;

    /**
     * @var bool Whether or not to set this input to the closest value when the user blurs
     */
    static setClosestOnBlur = true;

    /**
     * @var bool Whether or not to reset the value of the list input when focusing. 
     *           If this is false, it will show the selected value in the list input 
     *           and the user can deselect it.
     */
    static resetListOnFocus = true;

    /**
     * @var bool Whether or not to hide the list input on blur
     */
    static hideListOnBlur = true;

    /**
     * @var bool Whether or not to hide the list when a user starts typing
     */
    static hideListOnChange = true;

    /**
     * @var bool Whether or not to populate the search results when a value is set
     */
    static populateSearchOnSet = true;

    /**
     * @var int An interval to debounce change events on, in case one bubbles up from multi-inputs
     */
    static debounceChangeThreshold = 150;

    /**
     * On construction, create sub-inputs and bind events
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.stringInput = new this.constructor.stringInputClass(config, "autofill", {placeholder: this.constructor.placeholder});
        this.stringInput.onInput((searchValue) => this.searchChanged(searchValue));
        this.stringInput.onFocus(async () => {
            let stringInputNode = await this.stringInput.getNode(),
                listInputNode = await this.listInput.getNode(),
                inputPosition = this.node.element.getBoundingClientRect(),
                left = inputPosition.x,
                top = inputPosition.y + inputPosition.height,
                width = inputPosition.width,
                positionElement = (l, t, w) => {
                    left = l;
                    top = t;
                    width = w;
                    listInputNode.css({width: `${w}px`, left: `${l}px`, top: `${t}px`});
                };

            // Since we can't rely on any sort of event listener to follow the text input if it moves,
            // we have to set an interval to reposition the list input to line up with the text input.
            this.repositionInterval = setInterval(() => {
                inputPosition = this.node.element.getBoundingClientRect();
                let thisLeft = inputPosition.x,
                    thisTop = inputPosition.y + inputPosition.height,
                    thisWidth = inputPosition.width;

                if (left !== thisLeft || top !== thisTop || width !== thisWidth) {
                    positionElement(thisLeft, thisTop, thisWidth);
                }
            }, 25);

            document.body.appendChild(listInputNode.render());
            positionElement(left, top, width);
            listInputNode.css({
                width: `${inputPosition.width}px`,
                left: `${inputPosition.x}px`,
                top: `${inputPosition.y + inputPosition.height}px`
            });

            this.listInput.show();
            if (this.constructor.resetListOnFocus) {
                this.listInput.setValue(null, false);
            }

            let onClickElsewhereHideList = (e) => {
                this.listInput.hide();
                window.removeEventListener("click", onClickElsewhereHideList, false);
                clearTimeout(this.searchDebounceTimer);
                clearInterval(this.repositionInterval);
            };

            window.addEventListener("click", onClickElsewhereHideList, false);
        });

        this.stringInput.onBlur(async (e) => {
            if (this.constructor.hideListOnBlur) {
                let onMouseUpHideList = (e2) => {
                    setTimeout(async () => {
                        this.listInput.hide();
                        document.body.removeChild(
                            (await this.listInput.getNode()).element
                        );
                    }, 100);
                    window.removeEventListener("mouseup", onMouseUpHideList, true);
                };
                window.addEventListener("mouseup", onMouseUpHideList, true);
                clearTimeout(this.searchDebounceTimer);
                clearInterval(this.repositionInterval);
            }
            if (this.constructor.setClosestOnBlur) {
                let searchValue = strip(this.stringInput.getValue()).toLowerCase();
                if (isEmpty(searchValue)) {
                    this.value = null;
                } else {
                    let options = await this.getOptions(),
                        optionValues = Object.getOwnPropertyNames(options);
                    optionValues.sort(
                        (a, b) =>
                            jaroWinkler(options[b].toLowerCase(), searchValue) -
                            jaroWinkler(options[a].toLowerCase(), searchValue)
                    );
                    this.value = optionValues[0];
                    this.stringInput.setValue(options[this.value], false);
                }
            }
        });

        this.listInput = new this.constructor.listInputClass(config, "list", {options: () => this.getOptions()});
        this.listInput.onChange(async () => {
            let options = await this.getOptions(),
                value = this.listInput.getValue();
            this.value = value;
            if (this.constructor.populateSearchOnSet) {
                this.stringInput.setValue(options[value]);
            }
            if (this.constructor.hideListOnChange) {
                clearTimeout(this.searchDebounceTimer);
                clearInterval(this.repositionInterval);
            }
            this.changed();
        });
        this.listInput.hide();
    }

    /**
     * Disable both inputs and clear timers when disabled
     */
    disable() {
        clearTimeout(this.searchDebounceTimer);
        this.stringInput.disable();
        this.listInput.disable();
        if (this.node !== undefined) {
            for (let selectedItem of this.node.findAll(E.getCustomTag("searchSelection"))) {
                selectedItem.find("button").disabled(true);
            }
        }
    }

    /**
     * Re-enable both inputs when enabled
     */
    enable() {
        this.stringInput.enable();
        this.listInput.enable();
        if (this.node !== undefined) {
            for (let selectedItem of this.node.findAll(E.getCustomTag("searchSelection"))) {
                selectedItem.find("button").disabled(false);
            }
        }
    }

    /**
     * Set options to the sub-object
     */
    setOptions(newOptions) {
        super.setOptions(newOptions);
        this.listInput.setValue([], false);
        this.listInput.setOptions(newOptions);
    }

    /**
     * Override both of these functions to do nothing
     */
    rebuildOptions() {}
    buildOptions() {}

    /**
     * Force value to come from memory
     */
    getValue() {
        return this.value;
    }

    /**
     * Sets the value in both inputs
     */
    setValue(value, triggerChange) {
        let result = super.setValue(value, false);
        if (this.stringInput !== undefined) {
            this.stringInput.setValue(value, false);
        }
        if (triggerChange) {
            this.changed();
        }
    }

    /**
     * This function is called when the search input is changed.
     * Debounces the search, waiting searchTimeout milliseconds.
     *
     * @param string $searchValue The value from the search input.
     */
    searchChanged(searchValue) {
        clearTimeout(this.searchDebounceTimer);
        this.searchDebounceTimer = setTimeout(() => {
            this.search(searchValue);
        }, this.constructor.searchTimeout);
    }

    /**
     * This is the callback fired when the searchChanged timer expires.
     *
     * @param string $searchValue The string input in the search field
     */
    async search(searchValue) {
        searchValue = strip(searchValue.toLowerCase());
        if (isEmpty(searchValue)) {
            await this.listInput.sortFilterOptions(
                (a, b) => a - b,
                (a, b) => false
            );
        } else {
            await this.listInput.sortFilterOptions(
                (a, b, va, vb) => {
                    return (
                        jaroWinkler(vb.toLowerCase(), searchValue) -
                        jaroWinkler(va.toLowerCase(), searchValue)
                    );
                },
                (key, value) => {
                    return (
                        1.0 - jaroWinkler(value.toLowerCase(), searchValue) >=
                        this.constructor.filterThreshold
                    );
                }
            );
        }
    }

    /**
     * Build node and bind selections and re-ordering
     */
    async build() {
        let node = await super.build();
        node.append(await this.stringInput.getNode());
        node.on("dragover", (e) => {
            e.preventDefault();
            e.stopPropagation();

            let targetItem = e.target,
                allItems = node.findAll(E.getCustomTag("searchSelection")),
                left = e.offsetX;

            for (let item of allItems) {
                item.removeClass("drop-left").removeClass("drop-right");
            }

            while (
                [
                    "ENFUGUE-SEARCH-LIST-SELECTION",
                    "ENFUGUE-SEARCH-LIST"
                ].indexOf(targetItem.tagName) === -1
            ) {
                left += targetItem.offsetLeft;
                targetItem = targetItem.parentElement;
            }

            if (
                targetItem.tagName == "ENFUGUE-SEARCH-LIST-SELECTION" &&
                !targetItem.classList.contains("dragged")
            ) {
                let droppedValue = targetItem.dataset.selectValue,
                    droppedItem = allItems
                        .filter(
                            (node) => node.data("select-value") === droppedValue
                        )
                        .shift(),
                    itemWidth = targetItem.offsetWidth,
                    dropRight = left > itemWidth / 2;

                if (dropRight) {
                    droppedItem.addClass("drop-right");
                } else {
                    droppedItem.addClass("drop-left");
                }
            }
        }).on("drop", (e) => {
            let targetItem = e.target,
                draggedValue = e.dataTransfer.getData("text/plain"),
                left = e.offsetX,
                allItems = node.findAll(E.getCustomTag("searchSelection"));

            for (let item of allItems) {
                item.removeClass("drop-left").removeClass("drop-right");
            }

            while (
                [
                    "ENFUGUE-SEARCH-LIST-SELECTION",
                    "ENFUGUE-SEARCH-LIST"
                ].indexOf(targetItem.tagName) === -1
            ) {
                left += targetItem.offsetLeft;
                targetItem = targetItem.parentElement;
            }

            if (targetItem.tagName == "ENFUGUE-SEARCH-LIST-SELECTION") {
                let droppedValue = targetItem.dataset.selectValue;
                if (draggedValue !== droppedValue) {
                    let draggedItem = allItems
                            .filter((node) => node.data("select-value") === draggedValue)
                            .shift(),
                        droppedItem = allItems
                            .filter((node) => node.data("select-value") === droppedValue)
                            .shift(),
                        draggedIndex = this.value.indexOf(draggedValue),
                        droppedIndex = this.value.indexOf(droppedValue),
                        draggedLabel = draggedItem.find("span").getText(),
                        droppedLabel = droppedItem.find("span").getText(),
                        itemWidth = targetItem.offsetWidth,
                        dropRight = left > itemWidth / 2;

                    node.remove(draggedItem);
                    this.value = this.value.filter(
                        (value) => value != draggedValue
                    );
                    if (!dropRight && draggedIndex < droppedIndex) {
                        droppedIndex--;
                    } else if (dropRight && draggedIndex > droppedIndex) {
                        droppedIndex++;
                    }

                    this.value = this.value
                        .slice(0, droppedIndex)
                        .concat([draggedValue])
                        .concat(this.value.slice(droppedIndex));
                    this.listInput.setValue(this.value, false);
                    node.insert(droppedIndex, draggedItem);
                    this.changed();
                }
            }
        });
        return node;
    }

    async changed() {
        let changeTime = (new Date()).getTime();
        if (this.lastChange !== undefined && changeTime - this.lastChange < this.constructor.debounceChangeThreshold) {
            return;
        }
        await super.changed();
        this.lastChange = changeTime;
    }
}

/**
 * This class extends the parent ListInput to allow for multiple selections
 */
class SearchListMultiInputView extends SearchListInputView {
    /**
     * @var bool Disable setting closest
     */
    static setClosestOnBlur = false;

    /**
     * @var bool Disable resetting the child list when focusing
     */
    static resetListOnFocus = false;

    /**
     * @var bool Disable hiding the list when blurring
     */
    static hideListOnBlur = false;

    /**
     * @var bool Disable populating the search field when set is called
     */
    static populateSearchOnSet = false;

    /**
     * @var bool Disable hiding the list on change events
     */
    static hideListOnChange = false;

    /**
     * @var class Override the list input class to use the Multi version
     */
    static listInputClass = SearchListMultiInputListView;

    /**
     * On constructor, add change listener
     */
    constructor(config, fieldName, fieldConfig) {
        super(config, fieldName, fieldConfig);
        this.onChange(async () => {
            let value = this.getValue(),
                options = await this.getOptions();
            if (isEmpty(value)) value = [];
            if (this.node !== undefined) {
                this.node.content(
                    ...value.map((valuePart) => {
                        let selection = E.searchSelection().data(
                                "select-value",
                                valuePart
                            ),
                            removeSelection = E.button().content("&times;"),
                            selectionName = E.span().content(
                                options[valuePart]
                            );

                        removeSelection.on("click", (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            this.value = this.value.filter(
                                (value) =>
                                    value !== selection.data("select-value")
                            );
                            this.listInput.setValue(this.value, false);
                            this.changed();
                        });

                        selection
                            .attr("draggable", "true")
                            .on("dragstart", (e) => {
                                selection.addClass("dragged");
                                let dt = e.dataTransfer;
                                dt.dropEffect = "move";
                                dt.effectAllowed = "move";
                                dt.setData("text/plain", valuePart);
                            })
                            .on("dragend", (e) => {
                                selection.removeClass("dragged");
                            });

                        return selection.content(
                            removeSelection,
                            selectionName
                        );
                    }),
                    await this.stringInput.getNode()
                );
            }
        });
    }

    /**
     * When building, add string input.
     */
    async build() {
        let node = await super.build();
        node.append(await this.stringInput.getNode());
        return node;
    }
}

export {
    EnumerableInputView,
    SelectInputView,
    ListInputView,
    ListMultiInputView,
    SearchListInputView,
    SearchListInputListView,
    SearchListMultiInputView
};
