/** @module controllers/common/layers */
import { isEmpty } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";
import { View } from "../../view/base.mjs";
import { ImageView } from "../../view/image.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import { ImageEditorPromptNodeOptionsFormView } from "../../forms/enfugue/image-editor.mjs";

const E = new ElementBuilder();

/**
 * This view holds the menu for an individual layer.
 */
class LayerOptionsView extends View {
    /**
     * @var string Tag name
     */
    static tagName = "enfugue-layer-options-view";

    /**
     * @var string Text to show when no options
     */
    static placeholderText = "No options available. When you select a layer with options, they will appear in this pane.";

    /**
     * Sets the form
     */
    async setForm(formView) {
        this.node.content(await formView.getNode());
    }

    /**
     * Resets the form
     */
    async resetForm() {
        this.node.content(E.div().class("placeholder").content(this.constructor.placeholderText));
    }

    /**
     * On first build, append placeholder
     */
    async build() {
        let node = await super.build();
        node.content(
            E.div().class("placeholder").content(this.constructor.placeholderText)
        );
        return node;
    }
}

/**
 * This view allows you to select between individual layers
 */
class LayersView extends View {
    /**
     * @var string Tag name
     */
    static tagName = "enfugue-layers-view";

    /**
     * @var string Text to show when no layers
     */
    static placeholderText = "No layers yet. Use the buttons above to add layers, drag and drop videos or images onto the canvas, or paste media from your clipboard.";

    /**
     * On construct, create toolbar
     */
    constructor(config) {
        super(config);
        this.toolbar = new ToolbarView(config);
    }

    /**
     * Empties the layers
     */
    async emptyLayers() {
        this.node.content(
            await this.toolbar.getNode(),
            E.div().class("placeholder").content(this.constructor.placeholderText)
        );
    }

    /**
     * Adds a layer
     */
    async addLayer(newLayer, resetLayers = false) {
        if (resetLayers) {
            this.node.content(
                await this.toolbar.getNode(),
                await newLayer.getNode()
            );
        } else {
            this.node.append(await newLayer.getNode());
            this.node.render();
        }
    }

    /**
     * On first build, append placeholder
     */
    async build() {
        let node = await super.build();
        node.content(
            await this.toolbar.getNode(),
            E.div().class("placeholder").content(this.constructor.placeholderText)
        );
        node.on("drop", (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
        return node;
    }
}

/**
 * This class represents an individual layer
 */
class LayerView extends View {
    /**
     * @var int Preview width
     */
    static previewWidth = 30;

    /**
     * @var int Preview height
     */
    static previewHeight = 30;

    /**
     * @var string tag name in the layer view
     */
    static tagName = "enfugue-layer-view";

    /**
     * On construct, store editor node and form
     */
    constructor(controller, editorNode, form) {
        super(controller.config);
        this.controller = controller;
        this.editorNode = editorNode;
        this.form = form;
        this.isActive = false;
        this.isVisible = true;
        this.isLocked = false;
        this.previewImage = new ImageView(controller.config, this.layerImage, false);
        this.editorNode.onResize(() => this.resized());
    }

    /**
     * @var default foreground style
     */
    get foregroundStyle() {
        return window.getComputedStyle(document.documentElement).getPropertyValue("--theme-color-primary");
    }

    /**
     * Gets the layer image
     */
    get layerImage() {
        let width = this.controller.images.width,
            height = this.controller.images.height,
            maxDimension = Math.max(width, height),
            scale = this.constructor.previewWidth / maxDimension,
            widthRatio = width / maxDimension,
            heightRatio = height / maxDimension,
            previewWidth = this.constructor.previewWidth * widthRatio,
            previewHeight = this.constructor.previewHeight * heightRatio,
            nodeState = this.editorNode.getState(true),
            scaledX = nodeState.x * scale,
            scaledY = nodeState.y * scale,
            scaledWidth = nodeState.w * scale,
            scaledHeight = nodeState.h * scale,
            canvas = document.createElement("canvas");

        this.lastCanvasWidth = width;
        this.lastCanvasHeight = height;
        this.lastNodeWidth = nodeState.w;
        this.lastNodeHeight = nodeState.h;
        this.lastNodeX = nodeState.x;
        this.lastNodeY = nodeState.y;

        canvas.width = previewWidth;
        canvas.height = previewHeight;

        let context = canvas.getContext("2d");

        context.fillStyle = this.foregroundStyle;
        context.fillRect(scaledX, scaledY, scaledWidth, scaledHeight);

        return canvas.toDataURL();
    }

    /**
     * Triggers re-rendering of preview image if needed
     */
    async resized() {
        let width = this.controller.images.width,
            height = this.controller.images.height,
            nodeState = this.editorNode.getState();

        if (width !== this.lastCanvasWidth ||
            height !== this.lastCanvasHeight ||
            nodeState.w !== this.lastNodeWidth ||
            nodeState.h !== this.lastNodeHeight ||
            nodeState.x !== this.lastNodeX ||
            nodeState.y !== this.lastNodeY
        ) {
            this.previewImage.setImage(this.layerImage);
        }
    }

    /**
     * Removes this layer
     */
    async remove() {
        this.controller.removeLayer(this);
    }

    /**
     * Enables/disables a layer
     */
    async setActive(isActive) {
        this.isActive = isActive;
        if (this.isActive) {
            this.addClass("active");
        } else {
            this.removeClass("active");
        }
    }

    /**
     * Hides/shows a layer
     */
    async setVisible(isVisible) {
        this.isVisible = isVisible;
        if (!isEmpty(this.hideShowLayer)) {
            let hideShowLayerIcon = this.isVisible ? "fa-solid fa-eye": "fa-solid fa-eye-slash";
            this.hideShowLayer.setIcon(hideShowLayerIcon);
        }
        if (this.isVisible) {
            this.editorNode.show();
        } else {
            this.editorNode.hide();
        }
    }

    /**
     * Locks.unlocks a layer
     */
    async setLocked(isLocked) {
        this.isLocked = isLocked;
        if (!isEmpty(this.lockUnlockLayer)) {
            let lockUnlockLayerIcon = this.isLocked ? "fa-solid fa-lock" : "fa-solid fa-lock-open";
            this.lockUnlockLayer.setIcon(lockUnlockLayerIcon);
        }
        if (this.isLocked) {
            this.editorNode.addClass("locked");
        } else {
            this.editorNode.removeClass("locked");
        }
    }

    /**
     * Gets the state of editor node and form
     */
    getState(includeImages = true) {
        return {
            ...this.editorNode.getState(includeImages),
            ...this.form.values,
            ...{
                "isLocked": this.isLocked,
                "isActive": this.isActive,
                "isVisible": this.isVisible,
            }
        };
    }

    /**
     * Sets the state of the editor node and form, then populates DOM
     */
    async setState(newState) {
        await this.editorNode.setState(newState);
        await this.form.setValues(newState);
        this.previewImage.setImage(this.layerImage);
    }

    /**
     * Sets the name
     */
    async setName(name) {
        if (this.node !== undefined) {
            this.node.find("span.name").content(name);
        }
    }

    /**
     * On build, populate DOM with known details and buttons
     */
    async build() {
        let node = await super.build();

        this.toolbar = new ToolbarView(this.config);

        let hideShowLayerText = this.isVisible ? "Hide Layer" : "Show Layer",
            hideShowLayerIcon = this.isVisible ? "fa-solid fa-eye": "fa-solid fa-eye-slash";

        this.hideShowLayer = await this.toolbar.addItem(hideShowLayerText, hideShowLayerIcon);

        let lockUnlockLayerText = this.isLocked ? "Unlock Layer" : "Lock Layer",
            lockUnlockLayerIcon = this.isLocked ? "fa-solid fa-lock" : "fa-solid fa-lock-open";

        this.lockUnlockLayer = await this.toolbar.addItem("Lock Layer", "fa-solid fa-lock-open");
        this.hideShowLayer.onClick(() => this.setVisible(!this.isVisible));
        this.lockUnlockLayer.onClick(() => this.setLocked(!this.isLocked));

        node.content(
                await this.hideShowLayer.getNode(),
                await this.lockUnlockLayer.getNode(),
                E.span().class("name").content(this.editorNode.name),
                await this.previewImage.getNode(),
                E.button().content("&times;").class("close").on("click", () => this.remove())
            )
            .attr("draggable", "true")
            .on("dragstart", (e) => {
                e.dataTransfer.effectAllowed = "move";
                this.controller.draggedLayer = this;
                this.addClass("dragging");
            })
            .on("dragleave", (e) => {
                this.removeClass("drag-target-below").removeClass("drag-target-above");
                if (this.controller.dragTarget === this) {
                    this.controller.dragTarget = null;
                }
            })
            .on("dragover", (e) => {
                if (this.controller.draggedLayer !== this) {
                    let dropBelow = e.layerY > e.target.getBoundingClientRect().height / 2;
                    if (dropBelow) {
                        this.removeClass("drag-target-above").addClass("drag-target-below");
                    } else {
                        this.addClass("drag-target-above").removeClass("drag-target-below");
                    }
                    this.controller.dragTarget = this;
                    this.controller.dropBelow = dropBelow;
                }
            })
            .on("dragend", (e) => {
                this.controller.dragEnd();
                this.removeClass("dragging").removeClass("drag-target-below").removeClass("drag-target-above");
                e.preventDefault();
                e.stopPropagation();
            })
            .on("click", (e) => {
                this.controller.activate(this);
            })
            .on("drop", (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

        return node;
    }
}

/**
 * The LayersController manages the layer menu and holds state for each layer
 */
class LayersController extends Controller {
    /**
     * Removes layers
     */
    removeLayer(layerToRemove, removeNode = true) {
        if (removeNode) {
            layerToRemove.editorNode.remove(false);
        }
        let layerIndex = this.layers.indexOf(layerToRemove);
        if (layerIndex === -1) {
            console.error("Couldn't find", layerToRemove);
            return;
        }
        this.layers = this.layers.slice(0, layerIndex).concat(this.layers.slice(layerIndex+1));
        if (this.layers.length === 0) {
            this.layersView.emptyLayers();
            this.layerOptions.resetForm();
        } else {
            this.layersView.node.remove(layerToRemove.node.element);
        }
        if (layerToRemove.isActive) {
            this.layerOptions.resetForm();
        }
    }

    /**
     * Fired when done dragging layers
     */
    dragEnd() {
        if (!isEmpty(this.draggedLayer) && !isEmpty(this.dragTarget) && this.draggedLayer !== this.dragTarget) {
            this.draggedLayer.removeClass("dragging");
            this.dragTarget.removeClass("drag-target-above").removeClass("drag-target-below");

            let layerIndex = this.layers.indexOf(this.draggedLayer),
                targetIndex = this.layers.indexOf(this.dragTarget);
            
            if (targetIndex > layerIndex) {
                targetIndex--;
            }
            if (!this.dropBelow) {
                targetIndex++;
            }

            if (targetIndex !== layerIndex) {
                // Re-order on canvas (inverse)
                this.images.reorderNode(targetIndex, this.draggedLayer.editorNode);

                // Re-order in memory
                this.layers = this.layers.filter(
                    (layer) => layer !== this.draggedLayer
                );
                this.layers.splice(targetIndex, 0, this.draggedLayer);

                // Re-order in DOM
                this.layersView.node.remove(this.draggedLayer.node);
                this.layersView.node.insert(targetIndex + 1, this.draggedLayer.node);
                this.layersView.node.render();
            }
        }
        this.draggedLayer = null;
        this.dragTarget = null;
    }

    /**
     * Gets the state of all layers.
     */
    getState(includeImages = true) {
        return {
            "layers": this.layers.map((layer) => layer.getState(includeImages))
        }
    }

    /**
     * Gets the default state on init.
     */
    getDefaultState() {
        return {
            "layers": []
        };
    }

    /**
     * Sets the state from memory/file
     */
    async setState(newState) {
        this.emptyLayers();
        if (!isEmpty(newState.layers)) {
            for (let layer of newState.layers) {
                await this.addLayerByState(layer);
            }
            this.activateLayer(this.layers.length-1);
        }
    }

    /**
     * Adds a layer by state
     */
    async addLayerByState(layer, node = null) {
        let addedLayer;
        switch (layer.classname) {
            case "ImageEditorPromptNodeView":
                addedLayer = await this.addPromptLayer(false, node, layer.name);
                break;
            default:
                console.error(`Unknown layer class ${layer.classname}, skipping.`);
        }
        if (!isEmpty(addedLayer)) {
            await addedLayer.setState(layer);
        }
        return addedLayer;
    }

    /**
     * Empties layers
     */
    async emptyLayers() {
        this.layers = [];
        this.layersView.emptyLayers();
        this.layerOptions.resetForm();
    }

    /**
     * Activates a layer by index
     */
    async activateLayer(layerIndex) {
        if (layerIndex === -1) {
            return;
        }
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].setActive(i === layerIndex);
        }
        this.layerOptions.setForm(this.layers[layerIndex].form);
    }

    /**
     * Activates a layer by layer
     */
    activate(layer) {
        return this.activateLayer(
            this.layers.indexOf(layer)
        );
    }

    /**
     * Adds a layer
     */
    async addLayer(newLayer, activate = true) {
        // Bind editor node events
        newLayer.editorNode.onNameChange((newName) => {
            newLayer.setName(newName, false);
        });
        newLayer.editorNode.onClose(() => {
            this.removeLayer(newLayer, false);
        });
        this.layers.push(newLayer);
        await this.layersView.addLayer(newLayer, this.layers.length === 1);
        if (activate) {
            this.activateLayer(this.layers.length-1);
        }
    }

    /**
     * Adds an image layer
     */
    async addImageLayer(imageData, name) {
        let imageNode = await this.images.addImageNode(imageData, name);
    }

    /**
     * Adds a prompt layer
     */
    async addPromptLayer(activate = true, promptNode = null, name = "Prompt") {
        if (isEmpty(promptNode)) {
            promptNode = await this.images.addPromptNode(name);
        }

        let promptForm = new ImageEditorPromptNodeOptionsFormView(this.config),
            promptLayer = new LayerView(this, promptNode, promptForm);

        promptForm.onSubmit((values) => {
            promptNode.setPrompts(values.prompt, values.negativePrompt);
        });

        await this.addLayer(promptLayer, activate);
        
        return promptLayer;
    }

    /**
     * Prompts for an image then adds a layer
     */
    async promptAddImageLayer() {
        let imageToLoad;
        try {
            imageToLoad = await promptFiles();
        } catch(e) { }
        if (!isEmpty(imageToLoad)) {
            this.application.loadFile(imageToLoad, truncate(imageToLoad.name, 16));
        }
    }

    /**
     * Gets the layer corresponding to a node on the editor
     */
    getLayerByEditorNode(node) {
        return this.layers.filter((layer) => layer.editorNode === node).shift();
    }

    /**
     * After copying a node, adds a layer
     */
    async addCopiedNode(newNode, previousNode) {
        let existingLayer = this.getLayerByEditorNode(previousNode),
            existingLayerState = existingLayer.getState(),
            newNodeState = newNode.getState();

        await this.addLayerByState({...existingLayerState, ...newNodeState}, newNode);

        this.activateLayer(this.layers.length-1);
    }

    /**
     * On initialize, add menus to view
     */
    async initialize() {
        // Initial layers state
        this.layers = [];
        this.layerOptions = new LayerOptionsView(this.config);
        this.layersView = new LayersView(this.config);

        // Add layer tools
        let imageLayer = await this.layersView.toolbar.addItem("Image/Video", "fa-regular fa-image"),
            scribbleLayer = await this.layersView.toolbar.addItem("Draw Scribble", "fa-solid fa-pencil"),
            promptLayer = await this.layersView.toolbar.addItem("Region Prompt", "fa-solid fa-text-width");

        imageLayer.onClick(() => this.promptAddImageLayer());
        scribbleLayer.onClick(() => this.addScribbleLayer());
        promptLayer.onClick(() => this.addPromptLayer());

        // Add layer options
        this.application.container.appendChild(await this.layerOptions.render());
        this.application.container.appendChild(await this.layersView.render());

        // Register callbacks for image editor
        this.images.onNodeFocus((node) => {
            this.activate(this.getLayerByEditorNode(node));
        });
        this.images.onNodeCopy((newNode, previousNode) => {
            this.addCopiedNode(newNode, previousNode);
        });
    }
};

export { LayersController };
