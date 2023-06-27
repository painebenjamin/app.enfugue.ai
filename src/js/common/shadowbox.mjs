import { ElementBuilder } from '../base/builder.mjs';
import { Loader } from '../base/loader.mjs';

const E = new ElementBuilder();

class Shadowbox {
    constructor(className) {
        this.className = className;

        this.container = E.div()
            .addClass(className)
            .on('click', () => this.hide());
        this.close = E.button()
            .addClass(`${className}-close`)
            .content('×')
            .on('click', () => this.hide());
        this.contentContainer = E.div()
            .addClass(`${className}-content`)
            .on('click', (e) => e.stopPropagation());

        Loader.done(() => {
            document.body.appendChild(this.container.render());
        });
    }

    show(content) {
        this.container
            .content(this.contentContainer.content(this.close, content))
            .addClass('active');
    }

    hide() {
        this.container.removeClass('active').empty();
    }
}

export { Shadowbox };
