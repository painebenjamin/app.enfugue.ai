/** @module controller/sidebar/03-inpainting */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { ScribbleView } from "../../view/scribble.mjs";
import { ToolbarView } from "../../view/menu.mjs";
import { InpaintingFormView } from "../../forms/enfugue/inpainting.mjs";

/**
 * Register controller to add to sidebar and manage state
 */
class InpaintingController extends Controller {
    /**
     * When asked for state, return data from form
     */
    getState(includeImages = true) {
        return { 
            "inpainting": {
                "options": this.inpaintForm.values,
                "mask": includeImages ? this.scribbleView.src : null,
            }
        };
    }

    /**
     * Get default state
     */
    getDefaultState() {
        return {
            "inpainting": {
                "options": {
                    "outpaint": true,
                    "inpaint": false,
                    "croppedInpaint": true,
                    "croppedInpaintFeather": 32
                }
            }
        };
    }

    /**
     * Set state in the inpainting form
     */
    setState(newState) {
        if (!isEmpty(newState.inpainting)) {
            if (!isEmpty(newState.inpainting.options)) {
                this.inpaintForm.setValues(newState.inpainting.options).then(() => this.inpaintForm.submit());
            }
            if (!isEmpty(newState.inpainting.mask)) {
                let image = new Image();
                image.onload = () => {
                    this.scribbleView.setMemory(image);
                }
                image.src = newState.inpainting.mask;
            }
        }
    }

    /**
     * Prepares a menu (either in the header or standalone)
     */
    async prepareMenu(menu) {
        let pencilShape = await menu.addItem("Toggle Pencil Shape", "fa-regular fa-square", "q"),
            pencilErase = await menu.addItem("Toggle Eraser", "fa-solid fa-eraser", "s"),
            pencilClear = await menu.addItem("Clear Canvas", "fa-solid fa-delete-left", "l"),
            pencilFill = await menu.addItem("Fill Canvas", "fa-solid fa-fill-drip", "v"),
            pencilIncrease = await menu.addItem("Increase Pencil Size", "fa-solid fa-plus", "i"),
            pencilDecrease = await menu.addItem("Decrease Pencil Size", "fa-solid fa-minus", "d"),
            hideMask = await menu.addItem("Toggle Mask Visibility", "fa-solid fa-eye", "y"),
            lockMask = await menu.addItem("Toggle Mask Locked/Unlocked", "fa-solid fa-lock", "k");

        pencilShape.onClick(() => {
            if (this.scribbleView.shape === "circle") {
                this.scribbleView.shape = "square";
                pencilShape.setIcon("fa-regular fa-circle");
            } else {
                this.scribbleView.shape = "circle";
                pencilShape.setIcon("fa-regular fa-square");
            }
        });
        pencilErase.onClick(() => {
            if (this.scribbleView.isEraser) {
                this.scribbleView.isEraser = false;
                pencilErase.setIcon("fa-solid fa-eraser");
            } else {
                this.scribbleView.isEraser = true;
                pencilErase.setIcon("fa-solid fa-pencil");
            }
        });
        pencilClear.onClick(() => { this.scribbleView.clearMemory(); });
        pencilFill.onClick(() => { this.scribbleView.fillMemory(); });
        pencilIncrease.onClick(() => { this.scribbleView.increaseSize(); });
        pencilDecrease.onClick(() => { this.scribbleView.decreaseSize(); });
        hideMask.onClick(() => {
            if (this.scribbleView.hidden) {
                this.scribbleView.show();
                hideMask.setIcon("fa-solid fa-eye");
            } else {
                this.scribbleView.hide();
                hideMask.setIcon("fa-solid fa-eye-slash");
            }
        });
        lockMask.onClick(() => {
            if (this.scribbleView.hasClass("locked")) {
                this.scribbleView.removeClass("locked");
                lockMask.setIcon("fa-solid fa-lock");
            } else {
                this.scribbleView.addClass("locked");
                lockMask.setIcon("fa-solid fa-unlock");
            }
        });
    }

    /**
     * Resizes the mask to the engine width
     */
    resize() {
        this.scribbleView.resizeCanvas(
            this.images.width,
            this.images.height
        );
    }

    /**
     * On initialize, build sub controllers and add DOM nodes
     */
    async initialize() {
        this.scribbleView = new ScribbleView(
            this.config,
            this.engine.width,
            this.engine.height
        );
        this.scribbleView.hide();
        let setMaskTimer;
        this.scribbleView.onDraw(() => {
            clearTimeout(setMaskTimer);
            setMaskTimer = setTimeout(() => {
                this.engine.mask = this.scribbleView.src;
            }, 100);
        });
        this.scribbleToolbar = new ToolbarView(this.config);
        this.scribbleToolbar.addClass("inpainting");
        this.scribbleToolbar.hide()

        await this.prepareMenu(this.scribbleToolbar);

        this.inpaintForm = new InpaintingFormView(this.config);
        this.inpaintForm.hide();
        this.inpaintForm.onSubmit((values) => {
            // Show/hide parts
            if (values.inpaint) {
                this.publish("inpaintEnabled");
                this.scribbleView.show();
                this.scribbleToolbar.show();
                this.engine.mask = this.scribbleView.src;
            } else {
                this.publish("inpaintDisabled");
                this.scribbleView.hide();
                this.scribbleToolbar.hide();
                this.engine.mask = null;
            }
            // Set engine values
            this.engine.outpaint = values.outpaint;
            this.engine.cropInpaint = values.cropInpaint;
            this.engine.inpaintFeather = values.inpaintFeather;
        });

        this.subscribe("engineWidthChange", () => this.resize());
        this.subscribe("engineHeightChange", () => this.resize());
        this.subscribe("layersChanged", (layers) => {
            if (isEmpty(layers)) {
                this.inpaintForm.hide();
                this.scribbleView.hide();
                this.scribbleToolbar.hide();
                this.engine.mask = null;
            } else {
                this.inpaintForm.show();
                if (this.inpaintForm.values.inpaint) {
                    this.engine.mask = this.scribbleView.src;
                    this.scribbleView.show();
                    this.scribbleToolbar.show();
               }
            }
        });

        this.application.sidebar.addChild(this.inpaintForm);
        this.application.container.appendChild(await this.scribbleToolbar.render());
        (await this.application.images.getNode()).find("enfugue-image-editor-overlay").append(await this.scribbleView.getNode());
    }
}

export { InpaintingController as SidebarController };
