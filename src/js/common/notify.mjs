/** @module common/notify */
import { ElementBuilder } from '../base/builder.mjs';

const E = new ElementBuilder(),
    defaultDuration = 5000,
    hideDuration = 1000;

/**
 * This class allows for a very simple notification that displays on the screen then fades out and removes itself.
 */
class SimpleNotification {
    /**
     * Send the notification.
     */
    static notify(notificationText, duration) {
        let notification = E.span()
                .class('notification')
                .content(notificationText),
            container = E.div()
                .class('notification-container')
                .content(notification);
        document.body.appendChild(container.render());
        setTimeout(
            () => {
                container.css('opacity', 0);
                setTimeout(() => {
                    document.body.removeChild(container.element);
                }, hideDuration);
            },
            duration === undefined ? defaultDuration : duration
        );
    }
}

export { SimpleNotification };
