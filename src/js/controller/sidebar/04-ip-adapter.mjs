/** @module controlletr/sidebar/04-ip-adapter */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { IPAdapterFormView } from "../../forms/enfugue/ip-adapter.mjs";

/**
 * Extends the menu controller for state and init
 */
class IPAdapterController extends Controller {
    /**
     * Get data from the IP adapter form
     */
    getState(includeImages = true) {
        return { "ip": this.ipAdapterForm.values };
    }
    
    /**
     * Gets default state
     */
    getDefaultState() {
        return {
            "ip": {
                "ipAdapterModel": "default"
            }
        }
    }

    /**
     * Set state in the IP adapter form
     */
    setState(newState) {
        if (!isEmpty(newState.ip)) {
            this.ipAdapterForm.setValues(newState.ip).then(() => this.ipAdapterForm.submit());
        }
    };

    /**
     * On init, append form
     */
    async initialize() {
        this.ipAdapterForm = new IPAdapterFormView(this.config);
        this.ipAdapterForm.hide();
        this.ipAdapterForm.onSubmit(async (values) => {
            this.engine.ipAdapterModel = values.ipAdapterModel;
        });
        this.application.sidebar.addChild(this.ipAdapterForm);
        this.subscribe("layersChanged", (newLayers) => {
            if (newLayers.reduce((carry, item) => carry || item.imagePrompt, false)) {
                this.ipAdapterForm.show();
            } else {
                this.ipAdapterForm.hide();
            }
        });
    }
}

export { IPAdapterController as SidebarController };
