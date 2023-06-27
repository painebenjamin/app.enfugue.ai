/** @module controller/file/01-about */
import { ElementBuilder } from "../../base/builder.mjs";
import { isEmpty } from "../../base/helpers.mjs";
import { MenuController } from "../menu.mjs";
import { View } from "../../view/base.mjs";

const E = new ElementBuilder();

/**
 * The abouve view shows the same info as the footer as well as donate links
 */
class AboutView extends View {
    /**
     * @var string custom classname
     */
    static className = "about-view";

    /**
     * @var string The link to patreon
     */
    static patreonLink = "https://patreon.com/BenjaminPaine";
    
    /**
     * @var string The link to kofi
     */
    static kofiLink = "https://ko-fi.com/benjaminpaine";

    /**
     * On build, append content and links
     */
    async build() {
        let node = await super.build();
        node.content(
            E.div().class("center").content(
                E.img().src("/static/img/cloud-320.png")
            ),
            E.p().content(
                E.span().content("Enfugue is developed by "),
                E.a().href("mailto:benjamin@enfugue.ai").content("Benjamin Paine"),
                E.span().content(" and licensed under the "),
                E.a().href("https://www.gnu.org/licenses/agpl-3.0.html").target("_blank").content("GNU Affero General Public License (AGPL) v3.0"),
                E.span().content(".")
            ),
            E.p().content(
                E.span().content("Based on "),
                E.a().href("https://ommer-lab.com/research/latent-diffusion-models/").target("_blank").content("High-Resolution Image Synthesis with Latent Diffusion Models (A.K.A. LDM & Stable Diffusion)"),
                E.span().content(" by the Computer Vision & Learning Group, "),
                E.a().href("https://stability.ai/").target("_blank").content("Stability AI"),
                E.span().content(" and "),
                E.a().href("http://runwayml.com/").target("_blank").content("Runway"),
                E.span().content(", licensed under "),
                E.a().href("https://bigscience.huggingface.co/blog/bigscience-openrail-m").target("_blank").content("The BigScience (Creative ML) OpenRAIL-M License"),
                E.span().content(".")
            ),
            E.h2().content("Support"),
            E.p().content(
                E.span().content("If you would like to support the continued development of Enfugue, please visit one of the links below. Please do "),
                E.strong().content("not"),
                E.span().content(" do this if you are not in a financial position to do so comfortably.")
            ),
            E.p().content("Enfugue will never charge for updates or lock features behind donations."),
            E.div().class("donate-links").content(
                E.a().class("patreon").href(this.constructor.patreonLink).target("_blank").content("Support on Patreon"),
                E.a().class("ko-fi").href(this.constructor.kofiLink).target("_blank").content("Support on Ko-Fi")
            )
        );
        return node;
    }
}

/**
 * The about controller displays a little bit of source info and donation links
 */
class AboutController extends MenuController {
    /**
     * @var string The text to display
     */
    static menuName = "About";

    /**
     * @var string The icon to display
     */
    static menuIcon = "fa-solid fa-circle-info";

    /**
     * @var int The width of the about window
     */
    static aboutWindowWidth = 500;

    /**
     * @var int The height of the about window
     */
    static aboutWindowHeight = 550;

    /**
     * On click, reset state
     */
    async onClick() {
        if (isEmpty(this.aboutWindow)) {
            this.aboutWindow = await this.spawnWindow(
                "About Enfugue",
                await (new AboutView(this.config)).getNode(),
                this.constructor.aboutWindowWidth,
                this.constructor.aboutWindowHeight
            );
            this.aboutWindow.onClose(() => { this.aboutWindow = null; });
        } else {
            this.aboutWindow.focus();
        }
    }
};

export { AboutController as MenuController };
