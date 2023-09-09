import { uuid, createEvent, isEmpty, parseSelector } from './helpers.mjs';

let defaultTags = [
        'div',
        'span',
        'input',
        'br',
        'hr',
        'a',
        'p',
        'i',
        'strong',
        'em',
        'button',
        'input',
        'select',
        'option',
        'img',
        'h1',
        'h2',
        'h3',
        'h4',
        'ul',
        'ol',
        'li',
        'table',
        'tbody',
        'thead',
        'tr',
        'td',
        'th',
        'canvas',
        'label',
        'iframe',
        'form',
        'textarea',
        'pre',
        'fieldset',
        'legend',
        'code',
        'style',
        'link',
        'script'
    ],
    namespacedTags = {
        'http://www.w3.org/2000/svg': ['svg', 'path', 'rect', 'circle']
    },
    defaultAttributes = [
        'id',
        'type',
        'src',
        'action',
        'target',
        'placeholder',
        'href',
        'disabled',
        'selected',
        'alt',
        'value',
        'width',
        'height',
        'x',
        'y',
        'd',
        'cx',
        'cy',
        'r',
        'for',
        'name',
        'fill',
        'stroke',
        'rel'
    ],
    namespacedAttributes = { 'http://www.w3.org/2000/xlink': ['xref'] };

class FrameStacker {
    constructor() {
        this.callbacks = [];
        this.next = [];
        this.waiting = false;
    }

    requestFrame(callback) {
        this.callbacks.push(callback);
        if (this.waiting) return;
        window.requestAnimationFrame(() => this.fireCallbacks());
        this.waiting = true;
    }

    requestNextFrame(callback) {
        this.next.push(callback);
    }

    fireNextCallbacks() {
        for (let callback of this.next) {
            callback();
        }
        this.next = [];
    }

    fireCallbacks() {
        for (let callback of this.callbacks) {
            callback();
        }
        window.requestAnimationFrame(() => this.fireNextCallbacks());
        this.callbacks = [];
        this.waiting = false;
    }
}

const globalFrame = new FrameStacker();

class DOMElement {
    static assignDefaultAttributes = true;
    static assignNamespacedAttributes = true;
    static assignElementId = true;

    constructor(tagName, nameSpace) {
        this.nameSpace = nameSpace;
        this.elementId = uuid();
        this.tagName = tagName;
        this.attributes = {};
        this.namespacedAttributes = {};
        this.contentArray = [];
        this.elementClasses = [];
        this.events = {};
        this.styles = {};
        let i = 0,
            instance = this;

        if (this.constructor.assignDefaultAttributes) {
            for (i = 0; i < defaultAttributes.length; i++) {
                this[defaultAttributes[i]] = (function (defaultAttribute) {
                    return function (attributeValue) {
                        return instance.attr(defaultAttribute, attributeValue);
                    };
                })(defaultAttributes[i]);
            }
        }
        if (this.constructor.assignNamespacedAttributes) {
            for (let nameSpace in namespacedAttributes) {
                for (i = 0; i < namespacedAttributes[nameSpace].length; i++) {
                    this[namespacedAttributes[nameSpace][i]] = (function (
                        namespacedAttribute
                    ) {
                        return function (attributeValue) {
                            return instance.attr(
                                namespacedAttribute,
                                attributeValue,
                                nameSpace
                            );
                        };
                    })(namespacedAttributes[nameSpace][i]);
                }
            }
        }
        if (tagName === 'svg') {
            this.attributes['xmlns'] = 'http://www.w3.org/2000/svg';
            this.attributes['version'] = '1.1';
            this.attributes['xmlns:xlink'] = 'http://www.w3.org/1999/xlink';
        }
        if (this.constructor.assignElementId) {
            this.data({ 'element-id': this.elementId });
        }
    }

    static fromNode(node) {
        let elementNode = new DOMElement(node.tagName.toLowerCase()),
            nodeChildren = Array.from(node.children);

        for (let attributeName of node.getAttributeNames()) {
            if (['style'].indexOf(attributeName) != -1) {
                elementNode.attr(
                    attributeName,
                    node.getAttribute(attributeName)
                );
            }
        }

        for (let className of node.classList) {
            elementNode.addClass(className);
        }

        if (nodeChildren.length > 0) {
            elementNode.contentArray = nodeChildren.map((child) =>
                DOMElement.fromNode(child)
            );
        } else {
            let elementText = node.innerText;
            if (elementText.length > 0) {
                elementNode.content(elementText);
            }
        }

        elementNode.element = node;
        return elementNode;
    }

    editable() {
        return this.attr('contenteditable', 'true')
            .attr('tabindex', 0)
            .on('input', (e) => {
                this.contentArray[0] = this.element.innerText;
            });
    }

    empty() {
        this.contentArray = [];
        if (this.element !== undefined) {
            this.setDOMContent(this.contentArray);
        }
        return this;
    }

    matchesSelector(selector) {
        if (this.element !== undefined) {
            return this.element.matches(selector);
        } else {
            let selectorParts = parseSelector(selector);

            if (
                !isEmpty(selectorParts.tags) &&
                selectorParts.tags.indexOf(this.tagName) === -1
            ) {
                return false;
            }
            if (
                !isEmpty(selectorParts.ids) &&
                selectorParts.ids.indexOf(this.attr('id')) === -1
            ) {
                return false;
            }
            if (!isEmpty(selectorParts.classes)) {
                for (let classPart of selectorParts.classes) {
                    if (this.elementClasses.indexOf(classPart) == -1) {
                        return false;
                    }
                }
            }
            if (!isEmpty(selectorParts.attributes)) {
                for (let attributeName in selectorParts.attributes) {
                    if (
                        this.attr(attributeName) !==
                        selectorParts.attributes[attributeName]
                    ) {
                        return false;
                    }
                }
            }
            return true;
        }
        return false;
    }

    css(attributeName, attributeValue) {
        if (attributeValue === undefined) {
            if (typeof attributeName == 'object') {
                for (let actualAttributeName in attributeName) {
                    this.css(
                        actualAttributeName,
                        attributeName[actualAttributeName]
                    );
                }
                return this;
            }
            return this.styles[attributeName];
        } else if (attributeValue === null) {
            delete this.styles[attributeName];
        } else {
            if (typeof attributeValue === 'number') {
                if (
                    [
                        'left',
                        'right',
                        'top',
                        'bottom',
                        'width',
                        'height'
                    ].indexOf(attributeName) !== -1
                ) {
                    attributeValue += 'px';
                }
            }
            this.styles[attributeName] = attributeValue;
        }
        let element = this;
        this.attr(
            'style',
            Object.getOwnPropertyNames(this.styles)
                .map(
                    (cssProperty) =>
                        `${cssProperty}: ${element.styles[cssProperty]}`
                )
                .join('; ')
        );
        return this;
    }

    show() {
        if (this.display === undefined) {
            this.css('display', null);
        } else {
            this.css('display', this.display);
        }
        return this;
    }

    hide() {
        if (this.display === undefined && this.styles.display !== 'none') {
            this.display = this.styles.display;
        }
        this.css('display', 'none');
        return this;
    }

    attr(attributeName, attributeValue, nameSpace) {
        if (attributeValue === undefined) {
            if (nameSpace === undefined) {
                return this.attributes[attributeName];
            } else {
                return this.namespacedAttributes.hasOwnProperty(nameSpace)
                    ? this.namespacedAttributes[nameSpace][attributeName]
                    : undefined;
            }
        }
        if (attributeValue === null || attributeValue === false) {
            if (nameSpace === undefined) {
                if (this.attributes.hasOwnProperty(attributeName)) {
                    delete this.attributes[attributeName];
                    if (this.element !== undefined) {
                        this.removeDOMAttribute(attributeName, attributeValue);
                    }
                }
            } else {
                if (this.attributes.hasOwnProperty(nameSpace)) {
                    delete this.namespacedAttributes[namespace][attributeName];
                    if (this.element !== undefined) {
                        this.removeDOMAttributeNS(
                            attributeName,
                            attributeValue,
                            nameSpace
                        );
                    }
                }
            }
        } else {
            if (attributeValue === true) {
                attributeValue = attributeName;
            }
            if (nameSpace === undefined) {
                this.attributes[attributeName] = attributeValue;
                if (this.element !== undefined) {
                    this.setDOMAttribute(attributeName, attributeValue);
                }
            } else {
                if (!this.namespacedAttributes.hasOwnProperty(nameSpace)) {
                    this.namespacedAttributes[nameSpace] = {};
                }
                this.namespacedAttributes[nameSpace][attributeName] =
                    attributeValue;
                if (this.element !== undefined) {
                    this.setDOMAttributeNS(
                        attributeName,
                        attributeValue,
                        nameSpace
                    );
                }
            }
        }
        return this;
    }

    hasClass(className) {
        return this.elementClasses.indexOf(className) !== -1;
    }

    class(className) {
        for (let className of this.elementClasses) {
            this.removeClass(className);
        }
        return this.addClass(className);
    }

    addClass(className) {
        for (let splitClassName of className.split(' ')) {
            this.elementClasses.push(splitClassName);
            if (this.element !== undefined) {
                this.element.classList.add(splitClassName);
            }
        }
        return this;
    }

    removeClass(className) {
        this.elementClasses = this.elementClasses.filter(
            (cls) => cls !== className
        );
        if (this.element !== undefined) {
            this.element.classList.remove(className);
        }
        return this;
    }

    toggleClass(className) {
        if (this.hasClass(className)) {
            this.removeClass(className);
        } else {
            this.addClass(className);
        }
        return this;
    }

    data(attribute, value) {
        if (attribute === undefined) {
            return this.attributes.filter((kv) => kv.key.startsWith('data'));
        }

        if (typeof attribute === 'string') {
            if (value === undefined) {
                return this.attr('data-' + attribute);
            } else {
                this.attr('data-' + attribute, value);
            }
        } else {
            let attributeName;
            for (attributeName in attribute) {
                this.attr('data-' + attributeName, attribute[attributeName]);
            }
        }
        return this;
    }

    content() {
        this.empty();
        this.contentArray = Array.from(arguments);
        if (this.contentArray[0] === undefined) {
            this.contentArray = [];
        }
        if (this.element !== undefined) {
            this.setDOMContent(this.contentArray);
        }
        return this;
    }

    prepend() {
        this.contentArray = Array.from(arguments).concat(this.contentArray);
        if (this.element !== undefined) {
            this.setDOMContent(this.contentArray);
        }
        return this;
    }

    append() {
        this.contentArray = this.contentArray.concat(Array.from(arguments));
        if (this.element !== undefined) {
            this.appendDOMContent(
                this.contentArray[this.contentArray.length - 1]
            );
        }
        return this;
    }

    extend(contentArray) {
        let elementCount = contentArray.length;
        this.contentArray = this.contentArray.concat(contentArray);
        if (this.element !== undefined) {
            for (
                let i = this.contentArray.length - elementCount;
                i < this.contentArray.length;
                i++
            ) {
                this.appendDOMContent(this.contentArray[i]);
            }
        }
        return this;
    }

    insert(index, childElement) {
        this.contentArray = this.contentArray
            .slice(0, index)
            .concat([childElement])
            .concat(this.contentArray.slice(index));
        if (this.element !== undefined) {
            this.setDOMContent(this.contentArray);
        }
        return this;
    }

    insertAfter(after, childElement) {
        let contentIndex = this.getChildIndex(after);
        return this.insert(contentIndex + 1, childElement);
    }

    insertBefore(before, childElement) {
        let contentIndex = this.getChildIndex(before);
        return this.insert(contentIndex, childElement);
    }

    swapChildren(leftIndex, rightIndex) {
        let temporaryElement = this.contentArray[rightIndex];
        this.contentArray[rightIndex] = this.contentArray[leftIndex];
        this.contentArray[leftIndex] = temporaryElement;
        return this;
    }

    getChildIndex(childElement) {
        let elementId, indexId, index;

        if (childElement instanceof DOMElement) {
            elementId = childElement.elementId;
        } else {
            elementId = childElement.getAttribute('data-element-id');
        }

        if (elementId !== undefined) {
            for (let i = 0; i < this.contentArray.length; i++) {
                if (this.contentArray[i] instanceof DOMElement) {
                    indexId = this.contentArray[i].elementId;
                } else if (this.contentArray[i] instanceof Node) {
                    indexId =
                        this.contentArray[i].getAttribute('data-element-id');
                } else {
                    continue;
                }
                if (indexId !== undefined && indexId === elementId) {
                    index = i;
                    break;
                }
            }
        }

        return index;
    }

    getChildIndexById(elementId) {
        let indexId;
        for (let i = 0; i < this.contentArray.length; i++) {
            if (this.contentArray[i] instanceof DOMElement) {
                indexId = this.contentArray[i].elementId;
            } else {
                indexId = this.contentArray[i].getAttribute('data-element-id');
            }
            if (indexId === elementId) {
                return i;
            }
        }
        return null;
    }

    is(testElement) {
        let elementId;
        if (testElement instanceof DOMElement) {
            elementId = testElement.elementId;
        } else if (testElement instanceof Node) {
            elementId = testElement.getAttribute('data-element-id');
        }
        return elementId === this.elementId;
    }

    remove(childElement) {
        if (childElement === undefined) {
            if (this.element !== undefined) {
                this.element.remove();
                this.element = undefined;
            }
            return this;
        } else {
            let childIndex = this.getChildIndex(childElement);

            if (childIndex !== undefined) {
                let removedItem = this.contentArray[childIndex];
                this.contentArray = this.contentArray
                    .slice(0, childIndex)
                    .concat(this.contentArray.slice(childIndex + 1));
                if (this.element !== undefined) {
                    this.removeDOMContent(removedItem);
                }
                return this;
            }
        }
        console.error("Couldn't find child element", childElement);
        return this;
    }

    replace(childElement, replacementElement) {
        let childIndex = this.getChildIndex(childElement),
            element = this.element;

        if (childIndex !== undefined) {
            this.contentArray = this.contentArray
                .slice(0, childIndex)
                .concat([replacementElement])
                .concat(this.contentArray.slice(childIndex + 1));
            if (element !== undefined) {
                window.requestAnimationFrame(function () {
                    element.childNodes[childIndex].replaceWith(
                        replacementElement.render()
                    );
                });
            }
        } else if (this.contentArray.length > 0) {
            for (let child of this.contentArray) {
                child.replace(childElement, replacementElement);
            }
        }

        return this;
    }

    isEmpty() {
        return this.contentArray.length == 0;
    }

    children() {
        return this.contentArray;
    }

    getChild(index) {
        if (index < 0) {
            index = this.contentArray.length + index;
        }
        return this.contentArray[index];
    }

    getText() {
        return this.children()
            .map((child) => {
                if (typeof child == 'string') {
                    return child;
                } else if (child instanceof DOMElement) {
                    return child.getText();
                }
                return '';
            })
            .join('');
    }

    firstChild() {
        return this.contentArray[0];
    }

    lastChild() {
        return this.contentArray[this.contentArray.length - 1];
    }

    on(eventName, handler) {
        for (let splitEventName of eventName.split(',')) {
            if (this.element !== undefined) {
                this.setDOMEvent(splitEventName, handler);
            }
            if (this.events.hasOwnProperty(splitEventName)) {
                this.events[splitEventName].push(handler);
            } else {
                this.events[splitEventName] = [handler];
            }
        }
        return this;
    }

    off(eventName, handler) {
        for (let splitEventName of eventName.split(',')) {
            if (this.element !== undefined) {
                if (this.events[splitEventName] !== undefined) {
                    if (handler !== undefined) {
                        this.removeDOMEvent(splitEventName, handler);
                        this.events[splitEventName] = this.events[
                            splitEventName
                        ].filter(
                            (existingHandler) => existingHandler !== handler
                        );
                        return this;
                    } else {
                        for (let eventHandler of this.events[splitEventName]) {
                            this.removeDOMEvent(splitEventName, eventHandler);
                        }
                    }
                }
            }
            delete this.events[splitEventName];
        }
        return this;
    }

    val(newValue, triggerChange) {
        if (newValue === undefined) {
            if (this.element === undefined) {
                return undefined;
            }
            return this.element.value;
        } else {
            if (this.tagName == 'select') {
                for (let child of this.contentArray) {
                    if (child.value() === newValue) {
                        child.selected(true);
                    } else {
                        child.selected(false);
                    }
                }
            } else if (this.tagName === 'textarea') {
                if (this.element !== undefined) {
                    this.element.innerText = newValue;
                } else {
                    this.on('render', () => {
                        this.val(newValue).off('render');
                    });
                }
            } else if (this.tagName === 'input') {
                switch (this.attr('type')) {
                    case 'checkbox':
                        this.attr('checked', newValue);
                        if (this.element !== undefined) {
                            this.element.checked = newValue;
                        }
                        break;
                }
            }
            if (this.element !== undefined) {
                this.element.value = newValue;
            }
            this.value(newValue);
            if (triggerChange !== false) {
                this.trigger('change');
            }
        }
        return this;
    }

    focus() {
        this.element.focus();
        return this;
    }

    select() {
        this.element.select();
        return this;
    }

    trigger(eventName) {
        if (this.element !== undefined) {
            if (typeof eventName === 'string') {
                eventName = createEvent(eventName);
            }
            this.element.dispatchEvent(eventName);
        }
        return this;
    }

    removeDOMAttribute(attributeName) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot remove attributes.';
        }
        this.element.removeAttribute(attributeName);
        return this;
    }

    removeDOMAttributeNS(attributeName, nameSpace) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot remove attributes.';
        }
        this.element.removeAttributeNS(nameSpace, attributeName);
        return this;
    }

    setDOMAttribute(attributeName, attributeValue) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set attributes.';
        }
        this.element.setAttribute(attributeName, attributeValue);
        return this;
    }

    setDOMAttributeNS(attributeName, attributeValue, nameSpace) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set attributes.';
        }
        this.element.setAttributeNS(nameSpace, attributeName, attributeValue);
        return this;
    }

    addDOMClass(className) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set attributes.';
        }
        this.element.classList.add(className);
        return this;
    }

    setDOMEvent(eventName, eventHandler) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set event listeners.';
        }
        this.element.addEventListener(eventName, eventHandler);
        return this;
    }

    removeDOMEvent(eventName, oldHandler) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set event listeners.';
        }
        if (oldHandler !== undefined) {
            this.element.removeEventListener(eventName, oldHandler);
        }
    }

    appendDOMContent(content) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot append content.';
        }
        if (content instanceof DOMElement) {
            if (content.element === undefined) {
                content.render(this.element);
            }
            if (content.element instanceof DocumentFragment) {
                return this;
            }
            this.element.appendChild(content.element);
        } else {
            try {
                if (content instanceof DocumentFragment) {
                    return this;
                }
                this.element.appendChild(content);
            } catch (e) {
                if (content === null || content === undefined) {
                    throw (
                        'Cannot add null or undefined content to ' +
                        this.tagName
                    );
                } else {
                    console.error(
                        'Could not add content of type',
                        content.constructor.name,
                        content,
                        'to',
                        this
                    );
                    console.error(e);
                }
            }
        }
        return this;
    }

    removeDOMContent(content) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot remove content.';
        }
        if (content instanceof DOMElement) {
            if (content.element === undefined) {
                return this; // Already gone
            }
            if (content.element instanceof DocumentFragment) {
                throw 'Shadow elements cannot be detached; you must remove the parent element.';
            }
            this.element.removeChild(content.element);
        } else {
            this.element.removeChild(content);
        }
        return this;
    }

    setDOMContent(content) {
        if (this.element === undefined) {
            throw 'Element has not been rendered, cannot set content.';
        }

        let instance = this;
        globalFrame.requestFrame(function () {
            if (instance.element === undefined) return;

            while (instance.element.firstChild) {
                instance.element.removeChild(instance.element.firstChild);
            }

            if (!Array.isArray(content)) {
                content = [content];
            }

            if (content.length == 1) {
                if (typeof content[0] == 'string') {
                    if (
                        instance.element.tagName === 'CODE' ||
                        instance.element.tagName === 'PRE'
                    ) {
                        instance.element.innerText = content[0];
                    } else {
                        instance.element.innerHTML = content[0];
                    }
                    return instance.element;
                }
            }

            if (content.length >= 1) {
                let i = 0;
                for (i = 0; i < content.length; i++) {
                    if (content[i] instanceof DOMElement) {
                        if (content[i].element === undefined) {
                            content[i].render(instance.element);
                        }
                        if (content[i].element instanceof DocumentFragment) {
                            continue;
                        }
                        instance.element.appendChild(content[i].element);
                    } else {
                        try {
                            if (content[i] instanceof DocumentFragment) {
                                continue;
                            }
                            instance.element.appendChild(content[i]);
                        } catch (e) {
                            if (
                                content[i] === null ||
                                content[i] === undefined
                            ) {
                                console.error(
                                    'Could not add null or undefined content to',
                                    instance
                                );
                                console.trace(e)
                                throw (
                                    'Cannot add null or undefined content to ' +
                                    instance.tagName
                                );
                            } else {
                                console.error(
                                    'Could not add content of type',
                                    content[i].constructor.name,
                                    content[i],
                                    'to',
                                    instance
                                );
                                console.error(e);
                                console.trace(e);
                            }
                        }
                    }
                }
            }
        });

        return this;
    }

    createDOMElement() {
        return document.createElement(this.tagName);
    }

    render() {
        let firstRender = this.element === undefined;

        let renderedContentArray = [].concat(this.contentArray);

        if (this.element === undefined) {
            this.element = this.createDOMElement();
        }

        for (let i in renderedContentArray) {
            if (renderedContentArray[i] instanceof DOMElement) {
                renderedContentArray[i] = renderedContentArray[i].render(
                    this.element
                );
            }
        }

        let attributeName, className, eventName, eventHandler;

        for (attributeName in this.attributes) {
            this.setDOMAttribute(attributeName, this.attributes[attributeName]);
        }

        for (className of this.elementClasses) {
            this.addDOMClass(className);
        }

        for (eventName in this.events) {
            for (eventHandler of this.events[eventName]) {
                this.setDOMEvent(eventName, eventHandler);
            }
        }

        this.setDOMContent(renderedContentArray);

        if (firstRender) {
            this.trigger('render');
        }

        if (this.tagName === 'select' && this.element !== undefined) {
            globalFrame.requestNextFrame(() => {
                if (!isEmpty(this.attributes.value)) {
                    for (let child of this.element.childNodes) {
                        if (child.value == this.attributes.value) {
                            child.selected = true;
                            return;
                        }
                    }
                } else {
                    if(this.element.childNodes.length > 0) {
                        this.element.childNodes[0].selected = true;
                    }
                }
            });
        }
        if (this.tagName === 'input' && this.attributes.value !== undefined) {
            this.element.value = this.attributes.value;
        }

        return this.element;
    }

    async awaitRender() {
        if (this.element === undefined) {
            let self = this;
            this.render();

            return new Promise(function (resolve, reject) {
                self.on('ready', () => resolve(self.element));
            });
        }
        return Promise.resolve(this.element);
    }

    redraw() {
        this.element.innerHTML = this.element.innerHTML;
    }

    findDeep(selector) {
        for (let child of this.children()) {
            if (
                child instanceof DOMElement &&
                child.matchesSelector(selector)
            ) {
                return child;
            }
            let grandChild = child.findDeep(selector);
            if (grandChild !== undefined && grandChild !== null) {
                return grandChild;
            }
        }
        return null;
    }

    findWide(selector) {
        for (let child of this.children()) {
            if (
                child instanceof DOMElement &&
                child.matchesSelector(selector)
            ) {
                return child;
            }
        }
        for (let child of this.children()) {
            if (child instanceof DOMElement) {
                let grandChild = child.findWide(selector);
                if (grandChild !== null && grandChild !== undefined) {
                    return grandChild;
                }
            }
        }
        return null;
    }

    findAllDeep(selector) {
        let results = [];
        for (let child of this.children()) {
            if (child instanceof DOMElement) {
                if (child.matchesSelector(selector)) {
                    results.push(child);
                }
                results = results.concat(child.findAllDeep(selector));
            }
        }
        return results;
    }

    findAllWide(selector) {
        let results = [];
        for (let child of this.children()) {
            if (
                child instanceof DOMElement &&
                child.matchesSelector(selector)
            ) {
                results.push(child);
            }
        }
        for (let child of this.children()) {
            if (child instanceof DOMElement) {
                results = results.concat(child.findAllDeep(selector));
            }
        }
        return results;
    }

    find(selector) {
        return this.findWide(selector);
    }

    findAll(selector) {
        return this.findAllWide(selector);
    }

    scrollToBottom() {
        if (this.element !== undefined) {
            this.element.scrollTo(0, this.element.scrollHeight);
        }
        return this;
    }
}

class DOMElementList {
    static fromNodeList(nodeList) {
        return new DOMElementList(
            Array.from(nodeList).map((node) => {
                if (node instanceof Node) {
                    return DOMElement.fromNode(node);
                } else if (node instanceof DOMElement) {
                    return node;
                } else {
                    console.error('Cannot parse element', node);
                    throw new Error('Invalid argument.');
                }
            })
        );
    }

    constructor(elements) {
        this.elements = elements;
    }

    each(callback) {
        for (let child of this.children()) {
            callback.apply(child);
        }
    }

    children() {
        return this.elements;
    }
}

class ShadowDOMElement extends DOMElement {
    static assignElementId = false;

    createDOMElement() {
        if (isEmpty(this.parentElement)) {
            throw 'Shadow DOMs must be attached to an element, they cannot be instantiated on their own.';
        }
        try {
            return this.parentElement.attachShadow({ mode: 'open' });
        } catch (e) {
            if (this.parentElement.shadowRoot) {
                return this.parentElement.shadowRoot;
            } else {
                throw e;
            }
        }
    }

    setDOMAttribute(attributeName, attributeValue) {
        console.warn(
            'DOM attributes not supported on shadow elements, ignoring setting',
            attributeName,
            '=',
            attributeValue
        );
        return this;
    }

    setDOMAttributeNS(attributeName, attributeValue, nameSpace) {
        console.warn(
            'DOM attributes not supported on shadow elements, ignoring setting',
            `${nameSpace}:${attributeName}`,
            '=',
            attributeValue
        );
        return this;
    }

    setDOMEvent(eventName, eventHandler) {
        console.warn(
            'DOM events not supported on shadow elements, ignoring adding listener for',
            eventName
        );
        return this;
    }

    removeDOMEvent(eventName, oldHandler) {
        console.warn(
            'DOM events not supported on shadow elements, ignoring removing listener for',
            eventName
        );
        return this;
    }

    removeDOMAttribute(attributeName) {
        console.warn(
            'DOM attributes not supported on shadow elements, ignoring removing',
            attributeName
        );
        return this;
    }

    removeDOMAttributeNS(attributeName, nameSpace) {
        console.warn(
            'DOM attributes not supported on shadow elements, ignoring removing',
            `${nameSpace}:${attributeName}`
        );
        return this;
    }

    addDOMClass(className) {
        console.warn(
            'DOM classes not supported on shadow elements, ignoring adding',
            className
        );
        return this;
    }

    render(parentElement) {
        this.parentElement = parentElement;
        return super.render();
    }
}

let Builder = function (customTagList) {
    let i,
        ns,
        builderInstance = function (argument) {
            if (
                argument instanceof DOMElement ||
                argument instanceof DOMElementList
            ) {
                return argument;
            } else if (
                Array.isArray(argument) ||
                argument instanceof HTMLCollection ||
                argument instanceof NodeList
            ) {
                return DOMElementList.fromNodeList(argument);
            } else if (argument instanceof Node) {
                return DOMElement.fromNode(node);
            } else if (typeof argument === 'string') {
                let elementNodes = document.querySelectorAll(argument);
                if (elementNodes.length === 1) {
                    return DOMElement.fromNode(elementNodes[0]);
                } else {
                    return DOMElementList.fromNodeList(elementNodes);
                }
            } else {
                return this;
            }
        };

    builderInstance.createElement = function (tagName, nameSpace) {
        return new DOMElement(tagName, nameSpace);
    };

    builderInstance.createShadow = function () {
        return new ShadowDOMElement();
    };

    for (let customTagFunction in customTagList) {
        builderInstance[customTagFunction] = () =>
            builderInstance.createElement(customTagList[customTagFunction]);
    }

    builderInstance.getCustomTag = (tagName) =>
        isEmpty(customTagList) ? undefined : customTagList[tagName];

    for (i = 0; i < defaultTags.length; i++) {
        builderInstance[defaultTags[i]] = (function (tagName) {
            return function () {
                return builderInstance.createElement(tagName);
            };
        })(defaultTags[i]);
    }

    for (ns in namespacedTags) {
        for (i = 0; i < namespacedTags[ns].length; i++) {
            builderInstance[namespacedTags[ns][i]] = (function (
                tagName,
                nameSpace
            ) {
                return function () {
                    return builderInstance.createElement(tagName, nameSpace);
                };
            })(namespacedTags[ns][i], ns);
        }
    }

    return builderInstance;
};

export { Builder as ElementBuilder, DOMElement };
