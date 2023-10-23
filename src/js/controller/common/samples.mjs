/** @module controller/common/samples */
import { isEmpty } from "../../base/helpers.mjs";
import { Controller } from "../base.mjs";
import { SampleChooserView } from "../../view/samples.mjs";

/**
 * This controller allows for choosing between visible samples.
 */
class SamplesController extends Controller {
    /**
     * Sets samples
     */
    setSamples(samples, isAnimation) {
        this.sampleChooser.setSamples(samples);
    }

    /**
     * On initialize, add DOM nodes
     */
    async initialize() {
       // Create views
       this.sampleChooser = new SampleChooserView(this.config);

       // Add views to image editor
       let imageEditor = await this.images.getNode();
       imageEditor.append(await this.sampleChooser.getNode());
    }
}

export { SamplesController };
