/** @module view/table */
import { ElementBuilder } from '../base/builder.mjs';
import { SimpleNotification } from '../common/notify.mjs';
import { View } from './base.mjs';
import { StringInputView, ListInputView } from "./forms/input.mjs";
import { set, isEmpty, snakeCase, deepClone, stripHTML } from '../base/helpers.mjs';

const E = new ElementBuilder({
    tablePaging: 'enfugue-model-table-paging',
    tableSearching: 'enfugue-model-table-searching'
});

/**
 * The TableView is a simple table that shows in-memory data,
 * allowing for re-ordering.
 */
class TableView extends View {
    /**
     * @var string The tag of the element that will be built
     */
    static tagName = 'table';

    /**
     * @var array<string, string, callable> Buttons that will be added to every row. Classes can extend this.
     * @see TableView.addButton
     */
    static buttons = [];

    /**
     * @var string The text to show when no data exists.
     */
    static emptyRow = 'No Data';

    /**
     * @var bool Whether or not to apply the default sorting algorithm, or only callbacks.
     */
    static applyDefaultSort = true;

    /**
     * @var bool Whether or not sorting is supported. Default true.
     */
    static canSort = true;

    /**
     * @var bool Whether or not copying text in cells is supported. Default true.
     */
    static canCopy = true;

    /**
     * @var object<string, callable> An optional map of formatters for extending classes.
     */
    static columnFormatters = {};

    /**
     * @var array|object An optional initial list of columns
     */
    static columns = [];
    
    /**
     * @var array An optional initial sort
     */
    static sort = [];

    /**
     * Constructs a new TableView.
     * @param object $config The global configuration object.
     * @param ?array<object> $data Optional, the data at initialization. Can be set later.
     */
    constructor(config, data) {
        super(config);
        this.data = [];
        this.buttons = [];
        this.onSortCallbacks = [];
        this.applyDefaultSort = this.constructor.applyDefaultSort;
        this.sort = deepClone(this.constructor.sort);
        this.columnFormatters = deepClone(this.constructor.columnFormatters);
        this.columns = deepClone(this.constructor.columns);
        this.canCopy = this.constructor.canCopy;
        if (!isEmpty(data)) {
            this.setData(data, isEmpty(this.columns));
        }
    }

    /**
     * Adds a button to the button array - doesn't re-build table.
     * @param string $buttonName The text to put in the button tooltip.
     * @param string $buttonIcon The icon to display in the table.
     * @param callable $callback The function to call when clicked.
     */
    addButton(buttonName, buttonIcon, callback) {
        this.buttons.push({
            label: buttonName,
            icon: buttonIcon,
            click: callback
        });
    }

    /**
     * Adds a formatter to the formatter object after initialization
     * @param string $column The column name for the formatter.
     * @param callable $formatter The callback to pass the value of columns to.
     */
    setFormatter(column, formatter) {
        this.columnFormatters[column] = formatter;
    }

    /**
     * Adds an action to perform when sorting is done.
     * @param callable $callback A callback function that receives the new sorting details.
     * @see this.sort
     */
    onSort(callback) {
        this.onSortCallbacks.push(callback);
    }

    /**
     * Adds an action to perform when searching is done.
     * @param callable $callback A callback function that receives the new searching details.
     * @see this.search
     */
    onSearch(callback) {
        this.onSearchCallbacks.push(callback);
    }

    /**
     * Sets the data after initialization.
     * @param array<object> $data The data to put in the table.
     * @param bool $setColumns Whether or not to set the columns of the table to the columns in the data. Default true.
     */
    async setData(data, setColumns, sort=true) {
        this.data = data;
        if (setColumns !== false) {
            try {
                this.columns = Object.getOwnPropertyNames(this.data[0]);
                this.sort = this.sort.filter(
                    ([column, reverse]) => this.columns.indexOf(column) !== -1
                );
            } catch {
                this.columns = [];
                this.sort = [];
            }
        }
        if (sort) {
            await this.sortData(false);
        }
        if (this.node !== undefined) {
            let thead = this.node.find('thead'),
                tbody = this.node.find('tbody');
            if (setColumns !== false) {
                thead.content(await this.buildHeaderRow());
            }
            if (this.data.length === 0) {
                tbody.content(await this.buildEmptyRow());
            } else {
                tbody.empty();
                for (let datum of this.data) {
                    tbody.append(await this.buildDataRow(datum));
                }
                tbody.render();
            }
        }
    }

    /**
     * Adds a single row to the table.
     * @param object $datum The row to add.
     * @param bool setColumns Whether or not to update columns. Default FALSE.
     */
    addDatum(datum, setColumns) {
        this.data.push(datum);

        if (setColumns === true) {
            this.columns = Object.getOwnPropertyNames(datum);
            this.sort = this.sort.filter(
                ([column, reverse]) => this.columns.indexOf(column) !== -1
            );
        }

        if (this.node !== undefined) {
            let thead = this.node.find('thead'),
                tbody = this.node.find('tbody');

            if (setColumns === true) {
                thead.content(this.buildHeaderRow());
            }
            tbody.append(this.buildDataRow(datum));
        }
    }

    /**
     * Sets the columns after initialization.
     * Will re-draw when on the page.
     * @param array<string> $columns The new columns of the table view, or an object of columns to labels.
     */
    setColumns(columns) {
        this.columns = columns;
        this.sort = this.sort.filter(
            ([column, reverse]) =>
                Array.isArray(this.columns)
                    ? this.columns.indexOf(column) !== -1
                    : this.columns[column] !== undefined
        );
        if (this.node !== undefined) {
            this.node.find('thead').content(this.buildHeaderRow());
        }
    }

    /**
     * Sets the table to be empty.
     * Will re-draw when on the page.
     * @param ?array<string> $columns The new columns of the table view; optional.
     */
    setEmpty(columns) {
        if (columns !== undefined) {
            this.setColumns(columns);
        }
        this.data = [];
        if (this.node !== undefined) {
            this.node.find('tbody').content(this.buildEmptyRow());
        }
    }

    /**
     * This is the default sort callback.
     */
    async sortData(setData = true) {
        this.data.sort((left, right) => {
            for (let [sortColumn, reverse] of this.sort) {
                let leftValue = left[sortColumn],
                    rightValue = right[sortColumn],
                    compareResult;

                if (
                    typeof leftValue === 'string' &&
                    typeof rightValue === 'string'
                ) {
                    compareResult = leftValue.localeCompare(rightValue);
                }
                if (compareResult === undefined) {
                    try {
                        compareResult =
                            leftValue < rightValue
                                ? -1
                                : leftValue == rightValue
                                ? 0
                                : 1;
                    } catch (e) {
                        console.error(
                            'Error caught during sorting, likely due to incomparable types.'
                        );
                        console.error(e);
                        console.log(
                            'Left operand',
                            left,
                            'left value',
                            leftValue
                        );
                        console.log(
                            'Right operand',
                            right,
                            'right value',
                            rightValue
                        );
                        return 0;
                    }
                }
                if (compareResult !== 0) {
                    return compareResult * (reverse ? -1 : 1);
                }
            }
            return 0;
        });
        if (setData) {
            await this.setData(this.data, false, false);
        }
    }

    /**
     * This is called to trigger callbacks after sort is changed.
     */
    async sortChanged() {
        for (let callback of this.onSortCallbacks) {
            await callback(this.sort);
        }
        if (this.applyDefaultSort) {
            this.sortData();
        }
    }

    /**
     * This is called as necessary to rebuild the headers of the table.
     */
    async buildHeaderRow() {
        let headerRow = E.tr(),
            allHeaders = {},
            columnNames = Array.isArray(this.columns) ? this.columns : Object.getOwnPropertyNames(this.columns),
            columnLabels = Array.isArray(this.columns) ? this.columns : columnNames.map((column) => this.columns[column]);

        for (let i in columnNames) {
            let columnName = columnNames[i],
                columnLabel = columnLabels[i],
                canCopy = !!navigator.clipboard,
                tooltip =
                    columnLabel +
                    "<br/><em class='note'>Left-click to toggle sort." +
                    (canCopy ? ' Right-click to copy text.' : '') +
                    '</em>';

            let headerItem = E.th()
                    .content(columnLabel)
                    .class(snakeCase(columnName))
                    .data('tooltip', tooltip),
                headerSortIndex = -1;

            if (canCopy && this.canCopy) {
                headerItem.on('contextmenu', (e) => {
                    if (navigator.clipboard) {
                        navigator.clipboard.writeText(stripHTML(column));
                        SimpleNotification.notify('Copied to Clipboard', 1000);
                        e.preventDefault();
                        e.stopPropagation();
                    }
                });
            }

            for (let i = 0; i < this.sort.length; i++) {
                let [sortColumn, reverse] = this.sort[i];
                if (sortColumn === columnName) {
                    headerSortIndex = i;
                    break;
                }
            }

            allHeaders[columnName] = headerItem;

            if (headerSortIndex !== -1) {
                headerItem.addClass('sort');
                let reverse = this.sort[headerSortIndex][1];
                if (reverse) {
                    headerItem.addClass('sort-reverse');
                }
                headerItem.addClass(`sort-${headerSortIndex}`);
            }

            if (this.constructor.canSort) {
                headerItem.on('click', (e) => {
                    if (headerItem.hasClass('sort')) {
                        if (headerItem.hasClass('sort-reverse')) {
                            headerItem
                                .removeClass('sort')
                                .removeClass('sort-reverse');
                            // remove
                            this.sort = this.sort.filter(
                                ([sortColumn, reverse]) => sortColumn !== columnName
                            );
                            let sortIndex = 0;
                            for (let [sortColumn, reverse] of this.sort) {
                                for (let i = 0; i < 9; i++) {
                                    allHeaders[sortColumn].removeClass(`sort-${i}`);
                                }
                                allHeaders[sortColumn].addClass(
                                    `sort-${sortIndex++}`
                                );
                            }
                            this.sortChanged();
                        } else {
                            // reverse
                            headerItem.addClass('sort-reverse');
                            for (let i = 0; i < this.sort.length; i++) {
                                if (this.sort[i][0] === columnName) {
                                    this.sort[i][1] = true;
                                    break;
                                }
                            }
                            this.sortChanged();
                        }
                    } else {
                        // add, only allow 9 sort (thats enough)
                        if (this.sort.length < 9) {
                            headerItem
                                .addClass('sort')
                                .addClass(`sort-${this.sort.length}`);
                            this.sort.push([columnName, false]);
                            this.sortChanged();
                        }
                    }
                });
            }

            headerRow.append(headerItem);
        }

        for (let buttonConfig of this.constructor.buttons.concat(this.buttons)) {
            headerRow.append(
                E.th()
                    .content(buttonConfig.label)
                    .class(snakeCase(buttonConfig.label))
                    .addClass('button-column')
            );
        }

        return headerRow;
    }

    /**
     * This simply builds the empty row and returns it.
     */
    async buildEmptyRow() {
        return E.tr().content(
            E.td()
                .content(this.constructor.emptyRow)
                .attr(
                    'colspan',
                    this.columns.length +
                        this.constructor.buttons.length +
                        this.buttons.length
                )
        );
    }

    /**
     * The default formatter for values going into table cells
     */
    async defaultFormatter(value) {
        if (typeof value != 'string') {
            if (value === null) {
                return 'None';
            } else if (value === undefined) {
                return '';
            } else if (typeof value == 'boolean') {
                return value ? 'True' : 'False';
            } else if (value.toLocaleString !== undefined) {
                return value.toLocaleString();
            } else {
                return JSON.stringify(value); // Good luck!
            }
        }
        return value;
    }

    /**
     * Takes a datum and builds a table row.
     * @param object $datum The data to put in the table.
     */
    async buildDataRow(datum) {
        let dataRow = E.tr(),
            columns = Array.isArray(this.columns)
                ? this.columns
                : Object.getOwnPropertyNames(this.columns);

        for (let column of columns) {
            let value = datum[column];
            if (this.columnFormatters[column] == undefined) {
                value = await this.defaultFormatter(value);
            } else {
                value = await this.columnFormatters[column].call(this, value, datum);
            }
            let canCopy = !!navigator.clipboard,
                tooltip;

            if (typeof value === "string") {
                tooltip = value;
                if (canCopy) {
                    tooltip += "<br/><em class='note'>Right-click to copy text.</em>";
                }
            }

            let cell = E.td()
                .content(value)
                .class(snakeCase(column));
            
            if (!isEmpty(tooltip)) {
                cell.data('tooltip', tooltip);
                if (canCopy && this.canCopy) {
                    cell.on('contextmenu', (e) => {
                        if (navigator.clipboard) {
                            navigator.clipboard.writeText(stripHTML(value));
                            SimpleNotification.notify('Copied to Clipboard', 1000);
                            e.preventDefault();
                            e.stopPropagation();
                        }
                    });
                }
            }

            dataRow.append(cell);
        }

        for (let buttonConfig of this.constructor.buttons.concat(this.buttons)) {
            let buttonNode = E.button()
                .class('round')
                .content(E.i().class(buttonConfig.icon));
            buttonNode.data('tooltip', buttonConfig.label);
            buttonNode.on('click', () => buttonConfig.click.call(this, datum));
            dataRow.append(
                E.td()
                    .content(buttonNode)
                    .class(snakeCase(buttonConfig.label))
                    .addClass('button-column')
            );
        }

        return dataRow;
    }

    /**
     * Create and return the table node.
     */
    async build() {
        let node = await super.build(),
            headerRow = await this.buildHeaderRow(),
            thead = E.thead().content(headerRow),
            tbody = E.tbody();

        if (this.data.length === 0) {
            tbody.append(await this.buildEmptyRow());
        } else {
            for (let datum of this.data) {
                tbody.append(await this.buildDataRow(datum));
            }
        }
        return node.append(thead, tbody);
    }
}

/**
 * While not directly extending the TableView, this uses the TableView to show
 * a ModelBoundObject on the models object that receives standard pagination and filtering parameters,
 * basically automating the view for a database table.
 */
class ModelTableView extends View {
    /**
     * @var string The tag name of the element.
     */
    static tagName = 'enfugue-model-table';

    /**
     * @var int The number of rows to display at once.
     */
    static limit = 10;

    /**
     * @var int How many pages before and after the current page to display, when allowed.
     *          For example, if this is '2' and the user is on page '4', it will display:
     *          2 3 [4] 5 6
     */
    static pageWindowSize = 2;

    /**
     * @var int the number of milliseconds to wait for no input before starting the search
     */
    static searchTimeout = 250;

    /**
     * @var object Optional column formatters to pass to the table object at initialization.
     */
    static columnFormatters = {};

    /**
     * @var array|object An optional initial array of columns
     */
    static columns = [];
    
    /**
     * @var array An optional initial sort
     */
    static sortGroups = [];
    
    /**
     * @var array<object> Buttons that will be added to every row. Classes can extend this.
     * @see TableView.addButton
     */
    static buttons = [];

    /**
     * @var array<string> An array of columns that can be searched on.
     *                    Should end up being populated by string column types.
     */
    static searchFields = [];

    /**
     * Constructs a new TableView.
     * @param object $config The global configuration object.
     * @param class $modelObject The model object, an instance of /model/ModelBoundObject
     * @param ?object $initialFilters Optional, a list of intial filters for the table.
     */
    constructor(config, modelObject, initialFilters) {
        super(config);
        this.modelObject = modelObject;
        this.filter = isEmpty(initialFilters) ? {} : initialFilters;
        this.limit = this.constructor.limit;
        this.pageIndex = 0;
        this.customColumns = false;
        
        this.pages = {};
        this.table = new TableView(config);
        this.table.parent = this;
        this.table.applyDefaultSort = false;
        this.sortGroups = deepClone(this.constructor.sortGroups);
        this.searchFields = deepClone(this.constructor.searchFields);
        this.table.sort = this.sortGroups.map((sortPart) => [sortPart.column, sortPart.direction === "desc"]);
        this.table.onSort((sortGroups) => {
            this.tableSort = sortGroups;
        });
        if (!isEmpty(this.constructor.columnFormatters)) {
            for (let column in this.constructor.columnFormatters) {
                let formatter = this.constructor.columnFormatters[column];
                this.table.setFormatter(column, formatter);
            }
        }
        if (!isEmpty(this.constructor.columns)) {
            this.customColumns = true;
            this.table.setColumns(this.constructor.columns);
        }
        if (!isEmpty(this.constructor.buttons)) {
            for (let buttonConfiguration of this.constructor.buttons) {
                this.table.addButton(
                    buttonConfiguration.label,
                    buttonConfiguration.icon,
                    buttonConfiguration.click
                );
            }
        }
        this.paging = new ListInputView(config);
        this.paging.setOptions(['1']);
        this.paging.setValue('1', false);
        this.paging.onChange(() =>
            this.setPageIndex(parseInt(this.paging.getValue()) - 1)
        );
    }

    /**
     * Executes the query regardless of the stored state.
     * @param bool $includeMeta Whether or not the include metadata.
     * @return array Either an array including <array<data>, meta> or just array<data>
     */
    executeQuery(includeMeta) {
        let parameters = {"limit": this.limit, "offset": this.pageIndex * this.limit};
        if (!isEmpty(this.sortGroups)) {
            parameters["sort"] = this.sortGroups.map((sortGroup) => `${sortGroup.column}:${sortGroup.direction}`);
        }
        if (!isEmpty(this.searchFields) && !isEmpty(this.searchValue)) {
            parameters["ilike"] = this.searchFields.map((searchField) => `${searchField}:${encodeURIComponent(this.searchValue)}`);
        }
        return this.modelObject.query(this.filter, parameters, includeMeta);
    }

    /**
     * Gets the current page of data from memory.
     */
    get page() {
        return this.pages[this.pageIndex];
    }

    /**
     * Sets the current page of data in memory.
     */
    set page(pageData) {
        this.pages[this.pageIndex] = pageData;
    }

    /**
     * An alias for limit
     */
    get pageSize() {
        return this.limit;
    }

    /**
     * Returns either the ordered sort groups or an empty array
     */
    get tableSort() {
        return isEmpty(this.sortGroups) ? [] : this.sortGroups;
    }

    /**
     * Sets the sorting and executes the query
     */
    set tableSort(newSortGroups) {
        this.sortGroups = newSortGroups.map(([column, reverse]) => {
            return {
                column: column,
                direction: reverse ? 'desc' : 'asc'
            };
        });
        this.requery();
    }

    /**
     * Sets the search and executes the query
     */
    set tableSearch(newSearch) {
        this.searchValue = newSearch;
        this.requery();
    }

    /**
     * Gets the window of pages to display in the paging node.
     */
    get pageWindow() {
        let pageMinimum = Math.max(
                1,
                this.pageIndex + 1 - this.constructor.pageWindowSize
            ),
            pageMaximum = Math.min(
                pageMinimum + this.constructor.pageWindowSize * 2,
                this.count === 1 ? 1 : this.pageCount + 1
            ),
            pagesShown = pageMaximum - pageMinimum,
            pageShowMaximum = this.constructor.pageWindowSize * 2 + 1;
        
        if (pagesShown < pageShowMaximum && this.pageCount > pageShowMaximum) {
            let difference = pageShowMaximum - pagesShown;
            pageMinimum = Math.max(1, pageMinimum - difference + 1);
        }

        return new Array(pageMaximum - pageMinimum + 1)
            .fill(null)
            .map((_, i) => `${i + pageMinimum}`);
    }

    /**
     * Gets the actual options to display, including first and last.
     */
    get pageOptions() {
        let pageWindow = this.pageWindow,
            prependFirstPage = this.pageIndex > this.constructor.pageWindowSize,
            appendLastPage =
                this.pageIndex <
                this.pageCount - this.constructor.pageWindowSize;

        if (prependFirstPage) {
            pageWindow = ['1'].concat(pageWindow);
        }
        if (appendLastPage) {
            pageWindow.push(`${this.pageCount + 1}`);
        }
        return pageWindow;
    }

    /**
     * Gets the string that describes the rows show.
     */
    get rowRangeString() {
        if (this.count == 0) return '';
        let rowIndexStart = this.pageSize * this.pageIndex,
            rowIndexEnd = Math.min(
                this.count,
                (this.pageIndex + 1) * this.pageSize
            );

        return `${rowIndexStart + 1} — ${rowIndexEnd} of ${this.count}`;
    }

    /**
     * Sets a new page; this is called when page buttons are clicked.
     */
    async setPageIndex(pageIndex) {
        if (this.pageIndex !== pageIndex) {
            this.paging.disable();
            this.pageIndex = pageIndex;
            this.paging.setOptions(this.pageOptions);
            await this.table.setData(await this.getTableData(), !this.customColumns);
            if (this.node !== undefined) {
                this.node
                    .find(E.getCustomTag('tablePaging'))
                    .find('span')
                    .content(this.rowRangeString);
            }

            if (this.pageIndex > this.constructor.pageWindowSize + 1) {
                this.paging.addClass('include-first');
            } else {
                this.paging.removeClass('include-first');
            }

            if (
                this.pageIndex <
                this.pageCount - this.constructor.pageWindowSize - 1
            ) {
                this.paging.addClass('include-last');
            } else {
                this.paging.removeClass('include-last');
            }

            this.paging.enable();
        }
    }

    /**
     * Gets the table data, either from memory or loads it into memory then returns it.
     */
    async getTableData() {
        if (isEmpty(this.page)) {
            let data, meta;
            if (isEmpty(this.count)) {
                [data, meta] = await this.executeQuery(true);
                this.count = meta.count;
                this.pageCount = this.count > 1 
                    ? Math.floor((this.count - 1) / this.pageSize)
                    : this.count;
                this.paging.setOptions(this.pageOptions);
                this.paging.setValue('1', false);

                if (this.pageIndex > this.constructor.pageWindowSize + 1) {
                    this.paging.addClass('include-first');
                } else {
                    this.paging.removeClass('include-first');
                }

                if (this.pageIndex < this.pageCount - this.constructor.pageWindowSize - 1) {
                    this.paging.addClass('include-last');
                } else {
                    this.paging.removeClass('include-last');
                }
            } else {
                data = await this.executeQuery();
            }

            if (isEmpty(data)) {
                this.page = [];
            } else if (!Array.isArray(data)) {
                this.page = [data];
            } else {
                this.page = data;
            }
        }
        return this.page;
    }

    /**
     * Clears the page cache and re-queries (use after modifications)
     */
    async requery() {
        this.pages = {}
        this.count = null;
        await this.getTableData(); // Re-query for count
        let currentPageIndex = this.pageIndex;
        this.pageIndex = null;
        await this.setPageIndex(currentPageIndex); // Re-build paging
    }
    
    /**
     * Adds a button to the button array - doesn't re-build table.
     * @param string $buttonName The text to put in the button tooltip.
     * @param string $buttonIcon The icon to display in the table.
     * @param callable $callback The function to call when clicked.
     */
    addButton(buttonName, buttonIcon, callback) {
        this.table.addButton(buttonName, buttonIcon, callback);
    }

    /**
     * Sets specific headers instead of letting them auto-generate from attributes.
     * @param array<string> $newColumns The array of columns to include, or an object of columns and labels.
     */
    setColumns(newColumns){
        this.customColumns = newColumns;
        this.table.setColumns(newColumns);
    }
    
    /**
     * Adds a formatter to the table views' formatter object after initialization
     * @param string $column The column name for the formatter.
     * @param callable $formatter The callback to pass the value of columns to.
     */
    setFormatter(column, formatter) {
        this.table.setFormatter(column, formatter);
    }

    /**
     * Triggered when the search input has input; start/restart the timer to search.
     */
    debounceSearch(newText) {
        clearTimeout(this.searchTimer);
        this.searchTimer = setTimeout(() => {
            this.tableSearch = newText;
        }, this.constructor.searchTimeout);
    }

    /**
     * Builds the node, including the table node, paging node and descriptors.
     */
    async build() {
        let node = await super.build(),
            searching = E.tableSearching(),
            paging = E.tablePaging(),
            searchInputConfig = {
                "placeholder": "Start typing to search…"
            },
            searchInput = new StringInputView(this.config, "search", searchInputConfig);
        
        searchInput.onInput((searchText) => this.debounceSearch(searchText));
        searching.content(await searchInput.getNode());
        if (!isEmpty(this.modelObject)) {
            this.table.setData(await this.getTableData(), isEmpty(this.customColumns));
        }
        paging.content(
            await this.paging.getNode(),
            E.span().content(this.rowRangeString)
        );
        node.content(searching, await this.table.getNode(), paging);
        return node;
    }
}

export { TableView, ModelTableView };
