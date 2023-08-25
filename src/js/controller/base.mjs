/** @module controller/base */
import { isEmpty } from "../base/helpers.mjs";

/**
 * Allows a number of common operations controllers will want to perform.
 */
class Controller {
    /**
     * @var int The width of the 'confirm' form.
     */
    static confirmViewWidth = 500;

    /**
     * @var int The height of the 'confirm' form.
     */
    static confirmViewHeight = 200;

    /**
     * @param object $application All controllers are given a reference to the app.
     */
    constructor(application) {
        this.application = application;
    }

    /**
     * This is a stub, implementing classes can do anything they need to do after
     * first being built here.
     */
    async initialize() {
    
    }

    /**
     * @return models/enfugue.EnfugueModel A getter for the application model
     */
    get model() {
        return this.application.model;
    }

    /**
     * @return models/view/nodes/images.ImageEditorView A getter for the canvas
     */
    get images() {
        return this.application.images;
    }

    /**
     * @return object The global application configuration
     */
    get config() {
        return this.application.config;
    }

    /**
     * @return controllers/common/invocation.InvocationController Getter for the common invoke controller
     */
    get engine() {
        return this.application.engine;
    }

    /**
     * @return controllers/common/downloads.DownloadsController A getter for the common downloads controller
     */
    get downloads() {
        return this.application.downloads;
    }

    /**
     * Passes a notification to the app notification controller.
     * @param string $level The level - info, warn or error.
     * @param string $title The title of the notification.
     * @param string $message the content of the notification.
     */
    notify(level, title, message) {
        return this.application.notifications.push(level, title, message);
    }

    /**
     * Passes a window to the app windows controller.
     * @see view/nodes/windows.WindowsController
     */
    spawnWindow() {
        return this.application.windows.spawnWindow.apply(
            this.application.windows,
            Array.from(arguments)
        );
    }

    /**
     * This shorthand function passes a download to the download controller.
     * @param string $type The type of download - checkpoint, lora, inversion
     * @param string $url The url to download
     * @param string $filename The file name to save as in the correct folder
     */
    download(type, url, filename) {
        return this.downloads.download(type, url, filename);
    }

    /**
     * @param string $message The message to show in the confirmation box.
     * @return Promise A promise that is resolved with (true) if the user confirms, (false) otherwise.
     */
    confirm(message) {
        return this.application.confirm(message);
    }
    
    /**
     * Adds a callback for when an event is published.
     * @see base/publisher.subscribe
     */
    subscribe(eventName, callback) {
        this.application.subscribe(eventName, callback);
    }
    
    /**
     * Triggers callbacks for registered events.
     * @see base/publisher.subscribe
     */
    publish(eventName, payload = null) {
        this.application.publish(eventName, payload);
    }

    /**
     * Gets any state the controller would like to keep.
     * @param bool $includeImages Whether or not to include images for size considerations
     */
    getState(includeImages = true) {
        return {};
    }
    
    /**
     * Gets any default state the controller would like to set when resetting.
     */
    getDefaultState() {
        return {};
    }

    /**
     * Sets state; controllers can use as they wish
     */
    setState(state) { };
};

export { Controller };
