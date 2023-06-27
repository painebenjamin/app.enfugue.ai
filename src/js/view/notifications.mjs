import { View, ParentView } from './base.mjs';
import { ElementBuilder } from '../base/builder.mjs';
import { isEmpty } from '../base/helpers.mjs';

const E = new ElementBuilder(),
    hideDelay = 500,
    autoCloseDelay = 5000;

class NotificationView extends View {
    static tagName = 'enfugue-notification';

    constructor(config, title, message) {
        super(config);
        this.title = title;
        this.message = message;
    }

    close() {
        this.addClass('hiding');
        setTimeout(() => this.parent.removeChild(this), hideDelay);
    }

    async build() {
        let node = await super.build(),
            close = E.i().class('fas fa-times'),
            timer;
        node.content(
            E.h2().content(this.title),
            E.p().content(this.message),
            close
        );
        timer = setTimeout(() => this.close(), autoCloseDelay);
        node.on('mouseenter', () => {
            clearTimeout(timer);
        });
        close.on('click', () => this.close());
        return node;
    }
}

class ErrorNotificationView extends NotificationView {
    static className = 'error';
}

class WarnNotificationView extends NotificationView {
    static className = 'warn';
}

class InfoNotificationView extends NotificationView {
    static className = 'info';
}

class NotificationCenterView extends ParentView {
    static tagName = 'enfugue-notification-center';

    constructor(config) {
        super(config);
        for (let shortHandHelper of ['error', 'warn', 'info']) {
            this[shortHandHelper] = (title, message) =>
                this.push(shortHandHelper, title, message);
        }
    }

    push(level, title, message) {
        let messageString;
        if (message === null) {
            messageString = '[null]';
        } else if (message instanceof Error) {
            console.error(message);
            messageString = message.toString();
        } else if (typeof message === 'string') {
            messageString = message;
        } else if (typeof message === 'object') {
            if (!isEmpty(message.errors)) {
                messageString = `${message.errors[0].title}: ${message.errors[0].detail}`;
            } else if (!isEmpty(message.title)) {
                messageString = `${message.title}: ${message.detail}`;
            } else {
                messageString = JSON.stringify(message);
            }
        } else {
            messageString = `${message}`;
        }
        let className = {
            error: ErrorNotificationView,
            warn: WarnNotificationView,
            info: InfoNotificationView
        }[level.toLowerCase()];

        return this.addChild(className, title, messageString);
    }
}

export { NotificationCenterView };
