/** @module application/index */
import {
    isEmpty,
    getQueryParameters,
    getDataParameters,
    waitFor,
    createEvent
} from "../base/helpers.mjs";
import { Session } from "../base/session.mjs";
import { Publisher } from "../base/publisher.mjs";
import { TooltipHelper } from "../common/tooltip.mjs";
import { MenuView, SidebarView, ToolbarView } from "../view/menu.mjs";
import { StatusView } from "../view/status.mjs";
import { NotificationCenterView } from "../view/notifications.mjs";
import { WindowsView } from "../nodes/windows.mjs";
import { ImageView } from "../view/image.mjs";
import { Model } from "../model/enfugue.mjs";
import { View } from "../view/base.mjs";
import { FileNameFormView } from "../forms/enfugue/files.mjs";
import { StringInputView } from "../forms/input.mjs";
import { InvocationController } from "../controller/common/invocation.mjs";
import { ModelPickerController } from "../controller/common/model-picker.mjs";
import { ModelManagerController } from "../controller/common/model-manager.mjs";
import { DownloadsController } from "../controller/common/downloads.mjs";
import { AnimationsController } from "../controller/common/animations.mjs";
import { LogsController } from "../controller/common/logs.mjs";
import { AnnouncementsController } from "../controller/common/announcements.mjs";
import { HistoryDatabase } from "../common/history.mjs";
import { SimpleNotification } from "../common/notify.mjs";
import {
    CheckpointInputView,
    LoraInputView,
    LycorisInputView,
    InversionInputView,
    ModelPickerInputView,
    DefaultVaeInputView
} from "../forms/input.mjs";
import {
    ConfirmFormView,
    YesNoFormView
} from "../forms/confirm.mjs";
import {
    ImageEditorView,
    ImageEditorNodeView,
    ImageEditorImageNodeView
} from "../nodes/image-editor.mjs";

/**
 * Define a small view for a logout button
 */
class LogoutButtonView extends View {
    /**
     * @var string Change tagname to icon
     */
    static tagName = "i";

    /**
     * @var string Change classname to icon class
     */
    static className = "fa-solid fa-right-from-bracket logout";

    /**
     * Redirect to logout - which redirects to login
     */
    async logout() {
        window.location = "/logout";
    }

    /**
     * On build, bind click
     */
    async build() {
        let node = await super.build();
        node.on("click", () => this.logout());
        node.data("tooltip", "Logout");
        return node;
    }
};

/** 
 * The overall state container for the frontend. 
 */
class Application {
    /**
     * @var object The menu items to load. Will introspect controllers in these directories.
     */
    static menuCategories = {
        "file": "File"
    };

    /**
     * @var object Admin menus to load, only loads if window.enfugue.admin == true
     */
    static adminMenuCategories = {
        "models": "Models",
        "system": "System"
    };

    /**
     * @var int The width of the filename form
     */
    static filenameFormInputWidth = 400;

    /**
     * @var int The height of the filename form
     */
    static filenameFormInputHeight = 250;

    /**
     * @var array<int> The RGB colors (0-255) for the dynamic logo text shadow
     */
    static logoShadowRGB = [0, 0, 0];

    /**
     * @var float The starting opacity for the dynamic logo text shadow
     */
    static logoShadowOpacity = 0.5;

    /**
     * @var int The number of steps to generate for the dynamic logo text shadow
     */
    static logoShadowSteps = 10;

    /**
     * @var int The maximum offset of each shadow step for the dynamic logo text shadow
     */
    static logoShadowOffset = 2;

    /**
     * @var int The spread of each layer for the dynamic logo test shadow
     */
    static logoShadowSpread = 0;

    /**
     * @var int The width of the confirm form.
     */
    static confirmViewWidth = 500;

    /**
     * @var int The height of the confirm form.
     */
    static confirmViewHeight = 200;

    /**
     * Initializes the application.
     * @param {object} The configuration, an object of key-value pairs.
     */
    constructor(config) {
        this.config = config;
        this.publisher = new Publisher();
    }

    /**
     * Performs all the actions necessary to initialize the frontend.
     */
    async initialize() {
        this.tooltips = new TooltipHelper();
        this.container = document.querySelector(this.config.view.applicationContainer);
        if (isEmpty(this.container)) {
            console.error(`Couldn't find application configuration using selector ${this.config.view.applicationContainer}, abandoning initialization.`);
            return;
        }
        this.session = Session.getScope("enfugue");
        this.model = new Model(this.config);
        this.menu = new MenuView(this.config);
        this.sidebar = new SidebarView(this.config);
        this.toolbar = new ToolbarView(this.config);
        this.windows = new WindowsView(this.config);
        this.notifications = new NotificationCenterView(this.config);
        this.history = new HistoryDatabase(this.config.history.size, this.config.debug);
        this.images = new ImageEditorView(this);

        this.container.appendChild(await this.menu.render());
        this.container.appendChild(await this.sidebar.render());
        this.container.appendChild(await this.toolbar.render());
        this.container.appendChild(await this.images.render());
        this.container.appendChild(await this.windows.render());
        this.container.appendChild(await this.notifications.render());
        
        await this.startAnimations();
        await this.registerDynamicInputs();
        await this.registerDownloadsControllers();
        await this.registerAnimationsControllers();
        await this.registerModelControllers();
        await this.registerInvocationControllers();
        await this.registerMenuControllers();
        await this.registerSidebarControllers();
        await this.registerToolbarControllers();

        await this.startAutosave();
        await this.startAnnouncements();
        await this.startLogs();
        await this.startKeepalive();
        await this.registerLogout();

        window.onpopstate = (e) => this.popState(e);
        document.addEventListener("paste", (e) => this.onPaste(e));
        document.addEventListener("keypress", (e) => this.onKeyPress(e));
    }

    /**
     * Starts the logs controller which will read engine logs and display a limited
     * set of information on the screen, with a way to get more details.
     */
    async startLogs() {
        this.logs = new LogsController(this);
        await this.logs.initialize();
    }

    /**
     * Starts the announcements controller which will get necessary initialization
     * actions as well as versions from remote.
     */
    async startAnnouncements() {
        this.announcements = new AnnouncementsController(this);
        await this.announcements.initialize();
    }

    /**
     * Binds animations that should start immediately.
     */
    async startAnimations() {
        let headerLogo = document.querySelector("header h1");
        if (isEmpty(headerLogo)) {
            console.warn("Can't find header logo, not binding animations.");
            return;
        }
        this.animations = true;
        window.addEventListener("mousemove", (e) => {
            if (this.animations === false) return;
            let [x, y] = [
                    e.clientX / window.innerWidth,
                    e.clientY / window.innerHeight
                ],
                textShadowParts = [];
            
            for (let i = 0; i < this.constructor.logoShadowSteps; i++) {
                let [shadowDistanceX, shadowDistanceY] = [
                        x * (i + 1) * this.constructor.logoShadowOffset,
                        y * (i + 1) * this.constructor.logoShadowOffset
                    ],
                    shadowOpacity = this.constructor.logoShadowOpacity - (
                        (i / this.constructor.logoShadowSteps) * this.constructor.logoShadowOpacity
                    ),
                    shadowColor = `rgba(${this.constructor.logoShadowRGB.concat(shadowOpacity.toFixed(2)).join(',')})`;
                textShadowParts.push(`${shadowDistanceX}px ${shadowDistanceY}px ${this.constructor.logoShadowSpread}px ${shadowColor}`);
            }
            headerLogo.style.textShadow = textShadowParts.join(",");
        });
    }

    /**
     * Turns on animations
     */
    async enableAnimations() {
        if (this.animations) return;
        this.animations = true;
        this.session.setItem("animations", true);
        document.body.classList.remove("no-animations");
        this.publish("animationsEnabled");
    }

    /**
     * Turns off animations
     */
    async disableAnimations() {
        if (!this.animations) return;
        this.animations = false;
        this.session.setItem("animations", false);
        document.body.classList.add("no-animations");
        this.publish("animationsDisabled");
    }

    /**
     * Sets getters for dynamic inputs
     */
    async registerDynamicInputs() {
        if (isEmpty(window.enfugue) || !window.enfugue.admin) {
            // Remove other input
            delete DefaultVaeInputView.defaultOptions.other;
        }
        CheckpointInputView.defaultOptions = () => this.model.get("/checkpoints");
        LoraInputView.defaultOptions = () => this.model.get("/lora");
        LycorisInputView.defaultOptions = () => this.model.get("/lycoris");
        InversionInputView.defaultOptions = () => this.model.get("/inversions");
        ModelPickerInputView.defaultOptions = async () => {
            let allModels = await this.model.get("/model-options"),
                modelOptions = allModels.reduce((carry, datum) => {
                    let typeString = datum.type === "checkpoint"
                        ? "Checkpoint"
                        : "Preconfigured Model";
                    carry[`${datum.type}/${datum.name}`] = `<strong>${datum.name}</strong><em>${typeString}</em>`;
                    return carry;
                }, {});
            return modelOptions;
        };
    }

    /**
     * Registers the model picker controller
     */
    async registerModelControllers() {
        this.modelManager = new ModelManagerController(this);
        await this.modelManager.initialize();
        this.modelPicker = new ModelPickerController(this);
        await this.modelPicker.initialize();
    }

    /**
     * Registers the download manager
     */
    async registerDownloadsControllers() {
        this.downloads = new DownloadsController(this);
        await this.downloads.initialize();
    }

    /**
     * Creates the invocation engine manager.
     */
    async registerInvocationControllers() {
        this.engine = new InvocationController(this);
        await this.engine.initialize();
    }

    /**
     * Creates the animation manager (enable/disable animations.)
     */
    async registerAnimationsControllers() {
        this.animation = new AnimationsController(this);
        await this.animation.initialize();
    }

    /**
     * Returns the menu categories to import based on user context
     */
    getMenuCategories() {
        let menuCategories = {...this.constructor.menuCategories};
        if (!isEmpty(window.enfugue) && window.enfugue.admin === true) {
            menuCategories = {...menuCategories, ...this.constructor.adminMenuCategories };
        }
        menuCategories.help = "Help";
        return menuCategories;
    }

    /**
     * Registers menus and their controllers.
     */
    async registerMenuControllers() {
        let menuCategories = this.getMenuCategories();
        this.menuControllers = {};
        for (let menuCategoryName in menuCategories) {
            let menuCategoryLabel = menuCategories[menuCategoryName];
            this.menuControllers[menuCategoryName] = [];
            try {
                let menuCategoryModule = await import(`../controller/${menuCategoryName}/index.autogenerated.mjs`);
                let menuCategory = await this.menu.addCategory(menuCategoryLabel);
                for (let menuControllerPath of menuCategoryModule.Index) {
                    try {
                        let menuItemControllerModule = await import(`../controller/${menuCategoryName}/${menuControllerPath}`);
                        let menuItemControllerClass = menuItemControllerModule.MenuController;
                        if (isEmpty(menuItemControllerClass)) {
                            throw "Module does not provide a 'MenuController' export.";
                        }
                        if (!menuItemControllerClass.isDisabled()) {
                            let menuItem = await menuCategory.addItem(menuItemControllerClass.menuName, menuItemControllerClass.menuIcon);
                            let menuItemController = new menuItemControllerClass(this);
                            await menuItemController.initialize();
                            menuItem.onClick(() => menuItemController.onClick());
                            this.menuControllers[menuCategoryName].push(menuItemController);
                        }
                    } catch(e) {
                        console.warn("Couldn't import", menuCategoryName, "menu controller", menuControllerPath, e);
                    }
                }
            } catch(e) {
                console.warn("Couldn't register controllers for menu", menuCategoryName, e);
            }
        }
    }

    /**
     * Registers sidebar controllers.
     */
    async registerSidebarControllers() {
        let sidebarModule = await import("../controller/sidebar/index.autogenerated.mjs");
        this.sidebarControllers = [];
        for (let sidebarControllerPath of sidebarModule.Index) {
            let sidebarControllerModule = await import(`../controller/sidebar/${sidebarControllerPath}`);
            let sidebarControllerClass = sidebarControllerModule.SidebarController;
            if (isEmpty(sidebarControllerClass)) {
                throw "Module does not provide a 'SidebarController' export.";
            }
            let sidebarController = new sidebarControllerClass(this);
            await sidebarController.initialize();
            this.sidebarControllers.push(sidebarController);
        }
    }

    /**
     * Registers toolbar controllers.
     */
    async registerToolbarControllers() {
        let toolbarModule = await import("../controller/toolbar/index.autogenerated.mjs");
        this.toolbarControllers = [];
        for (let toolbarControllerPath of toolbarModule.Index) {
            let toolbarControllerModule = await import(`../controller/toolbar/${toolbarControllerPath}`);
            let toolbarControllerClass = toolbarControllerModule.ToolbarController;
            if (isEmpty(toolbarControllerClass)) {
                throw "Module does not provide a 'ToolbarController' export.";
            }
            let toolbarItem = await this.toolbar.addItem(toolbarControllerClass.menuName, toolbarControllerClass.menuIcon),
                toolbarItemController = new toolbarControllerClass(this);
            await toolbarItemController.initialize();
            toolbarItem.onClick(() => toolbarItemController.onClick());
            this.toolbarControllers.push(toolbarItemController);
        }
    }

    /**
     * Starts the process that periodically checks status.
     */
    async startKeepalive() {
        const statusInterval = this.config.model.status.interval || 10000,
              triggerStateChange = (newStatus) => {
                    if (newStatus === "ready") {
                        this.publish("engineReady");
                    } else if(newStatus === "busy") {
                        this.publish("engineBusy");
                    } else if(newStatus === "idle") {
                        this.publish("engineIdle");
                    } else {
                        console.warn("Unknown status", newStatus);
                    }
              };

        let header = document.querySelector("header");
        if (isEmpty(header)) {
            console.warn("No header found on page, not appending status view.");
            return;
        }

        let status = await this.model.get(),
            statusView = new StatusView(
                this.config,
                status
            ),
            statusViewNode = await statusView.getNode();

        if (!isEmpty(status.system) && !isEmpty(status.system.downloads) && status.system.downloads.active > 0) {
            this.downloads.checkStartTimer(); // Start the download timer if it's not already going
        }

        triggerStateChange(status.status);
        header.appendChild(statusViewNode.render());
        setInterval(async () => {
            try {
                let newStatus = await this.model.get();
                if (status.status !== newStatus.status) {
                    triggerStateChange(newStatus.status);
                }

                if (!isEmpty(status.system) && !isEmpty(status.system.downloads) && status.system.downloads.active > 0) {
                    this.downloads.checkStartTimer();
                }

                statusView.updateStatus(newStatus);
                status = newStatus;
            } catch {
                statusView.updateStatus("error");
            }
        }, statusInterval);

        let currentInvocations = await this.model.get("/invocation"),
            activeInvocation = null;
        
        for (let invocation of currentInvocations) {
            if (invocation.status === "processing") {
                activeInvocation = invocation;
                break;
            }
        }

        if (!isEmpty(activeInvocation)) {
            // There is an active invocation for this user at present
            if (!isEmpty(activeInvocation) && !isEmpty(activeInvocation.metadata) && !isEmpty(activeInvocation.metadata.tensorrt_build)) {
                this.notifications.push("info", "TensorRT Build in Progress", `You have a TensorRT engine build in progress for ${activeInvocation.metadata.tensorrt_build.model}. You'll receive a notification in this window when it is complete. The engine will remain unavailable until that time.`);
            } else {
                // Monitor it
                this.notifications.push("info", "Active Invocation Found", "You have an image currently being generated, beginning monitoring process.");
                this.engine.canvasInvocation(activeInvocation.uuid);
           }
           this.publish("invocationBegin", activeInvocation);
        }

        setInterval(async () => {
            try {
                let newInvocations = await this.model.get("/invocation"),
                    newActiveInvocation;

                for (let invocation of newInvocations) {
                    if (invocation.status === "processing") {
                        if (activeInvocation === null || activeInvocation.id !== invocation.id) {
                            newActiveInvocation = invocation;
                        }
                    }
                    if (!isEmpty(activeInvocation) && invocation.id === activeInvocation.id && invocation.status !== activeInvocation.status) {
                        if (invocation.status === "completed") {
                            this.publish("invocationComplete", invocation);
                        } else {
                            this.publish("invocationError", invocation);
                        }
                        activeInvocation = null;
                    }
                }
                
                if (!isEmpty(newActiveInvocation)) {
                    this.publish("invocationBegin", newActiveInvocation);
                    activeInvocation = newActiveInvocation;
                }
            } catch(e) {
                console.error(e);
            }
        }, statusInterval);
    }

    /**
     * Adds a button to logout if needed
     */
    async registerLogout() {
        if (!isEmpty(window.enfugue.user) && window.enfugue.user !== "noauth") {
            let logoutButton = new LogoutButtonView(this.config);
            document.querySelector("header").appendChild((await logoutButton.getNode()).render());
        }
    }

    /**
     * The autosave triggers every so often to save state
     */
    async startAutosave() {
        try {
            let existingAutosave = await this.history.getCurrentState();
            if (!isEmpty(existingAutosave)) {
                this.setState(existingAutosave);
                this.notifications.push("info", "Session Restored", "Your last autosaved session was successfully loaded.");
            }
            const autosaveInterval = this.config.model.autosave.interval || 30000;
            setInterval(() => this.autosave(), autosaveInterval);
        } catch(e) {
            console.error(e);
            this.notifications.push("warning", "History Disabled", "Couldn't open IndexedDB, history and autosave are disabled.");
        }
    }

    /**
     * Saves the current session data
     */
    async autosave(sendNotification = true) {
        try {
            await this.history.setCurrentState(this.getState());
            if (sendNotification) {
                SimpleNotification.notify("Session autosaved!", 2000);
            }
        } catch(e) {
            console.error("Couldn't autosave", e);
        }
    }

    /**
     * Adds a callback for when an event is published.
     * @see base/publisher.subscribe
     */
    subscribe(eventName, callback) {
        this.publisher.subscribe(eventName, callback);
    }
    
    /**
     * Removes a callback for when an event is published.
     * @see base/publisher.unsubscribe
     */
    unsubscribe(eventName, callback) {
        this.publisher.unsubscribe(eventName, callback);
    }
    
    /**
     * Triggers callbacks for registered events.
     * @see base/publisher.subscribe
     */
    async publish(eventName, payload = null) {
        this.publisher.publish(eventName, payload);
    }

    /**
     * Spawns a window with a confirmation message.
     * The promise returns true if the user confirms, false otherwise.
     * @param class $formClass The class of the confirm view, passed by implementing functions.
     * @param string $windowName The name of the window to display
     * @param string $message The message to display.
     * @param bool $closeOnSubmit Whether or not to close on submit. Default true.
     * @return Promise
     */
    spawnConfirmForm(formClass, windowName, message, closeOnSubmit = true, rejectOnClose = false) {
        return new Promise(async (resolve, reject) => {
            let resolved = false,
                confirmView = new formClass(this.config, message),
                confirmWindow;
            confirmView.onSubmit(() => {
                resolved = true;
                resolve(true);
                if (closeOnSubmit) {
                    confirmWindow.remove();
                }
            });
            confirmView.onCancel(() => {
                resolved = true;
                resolve(false);
                confirmWindow.remove();
            });
            confirmWindow = await this.windows.spawnWindow(
                windowName,
                confirmView, 
                this.constructor.confirmViewWidth,
                this.constructor.confirmViewHeight
            );
            confirmWindow.onClose(() => {
                if (!resolved && rejectOnClose) reject();
                else resolve(false);
                resolved = true;
            });
        });
    };

    /**
     * Spawns a window with a confirmation message.
     * The promise returns true if the user confirms, false otherwise.
     * @param string $message The message to display.
     * @param bool $closeOnSubmit Whether or not to close on submit. Default true.
     * @return Promise
     */
    confirm(message, closeOnSubmit = true) {
        return this.spawnConfirmForm(ConfirmFormView, "Confirm", message, closeOnSubmit);
    }
    
    /**
     * Spawns a window with a yes/no message.
     * The promise returns true if the user clicks 'yes', false otherwise.
     * Rejects (raises) if the user clicks 'x'.
     * @param string $message The message to display.
     * @param bool $closeOnSubmit Whether or not to close on submit. Default true.
     * @return Promise
     */
    yesNo(message, closeOnSubmit = true) {
        return this.spawnConfirmForm(YesNoFormView, "Yes or No", message, closeOnSubmit, true);
    }

    /**
     * Spawns a window asking for a filename, then downloads a file.
     * Uses Blob and Object URLs.
     * @param string $message The message to display before the input.
     * @param string $filename The default filename.
     * @param string $content The content of the blob.
     * @param string $fileType The file type of the content.
     * @param string $extension The file extension to append,
     */
    async saveAs(message, content, fileType, extension) {
        let blob = new Blob([content], {"type": fileType});
        return this.saveBlobAs(message, blob, extension);
    }
    
    /**
     * Spawns a window asking for a filename, then downloads a blob.
     * Uses Blob and Object URLs.
     * @param string $message The message to display before the input.
     * @param string $blob The blob.
     * @param string $extension The file extension to append.
     */
    async saveBlobAs(message, blob, extension) {
        if (!extension.startsWith(".")) {
            extension = `.${extension}`;
        }

        let fileURL = window.URL.createObjectURL(blob),
            fileNameForm = new FileNameFormView(this.config),
            fileNameWindow = await this.windows.spawnWindow(
                message,
                fileNameForm,
                this.constructor.filenameFormInputWidth,
                this.constructor.filenameFormInputHeight
            );
        fileNameForm.onCancel(() => fileNameWindow.close());
        fileNameForm.onSubmit((values) => {
            let filename = values.filename;
            if (filename.endsWith(extension)) {
                filename = filename.substring(0, filename.length - extension.length);
            }
            let linkElement = document.createElement("a");
            linkElement.setAttribute("download", `${filename}${extension}`);
            linkElement.href = fileURL;
            document.body.appendChild(linkElement);
            window.requestAnimationFrame(() => {
                linkElement.dispatchEvent(createEvent("click"));
                document.body.removeChild(linkElement);
                fileNameWindow.remove();
            });
        });
        fileNameWindow.onClose(() => window.URL.revokeObjectURL(fileURL));
    }

    /**
     * The onPaste handler reads the event and calls the relevant handler.
     * @param Event $e The paste event
     */
    async onPaste(e) {
        let pastedItems = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of pastedItems) {
            if (item.kind === "file") {
                let blob = item.getAsFile(),
                    reader = new FileReader();
                reader.onload = (e) => this.images.addImageNode(e.target.result, "Pasted Image");
                reader.readAsDataURL(blob);
            } else {
                item.getAsString((text) => this.onTextPaste(text));
            }
        }
    }
    
    /**
     * The onImagePaste handlers put an image on the canvas.
     * @param string $image The image source as a Data URI
     */
    async onImagePaste(image) {
        let imageView = new ImageView(this.config, image);
        await imageView.waitForLoad();
        this.images.addNode(
            ImageEditorImageNodeView, 
            "Pasted Image", 
            imageView,
            0,
            0,
            imageView.width,
            imageView.height
        );
    }

    /**
     * The onTextPaste handlers does nothing for now
     * @param string $text The text pasted to the window
     */
    async onTextPaste(text) {
        if (text.startsWith("<html>")) {
            // Ignore HTML paste
            return;
        }
        // TODO, maybe make a prompt node? Or automatically paste to prompt input?
    }

    /**
     * Gets all stateful controllers
     */
    getStatefulControllers() {
        let controllerArray = [this.modelPicker].concat(this.toolbarControllers).concat(this.sidebarControllers);
        for (let controllerName in this.menuControllers) {
            controllerArray = controllerArray.concat(this.menuControllers[controllerName]);
        }
        return controllerArray;
    }

    /**
     * Gets current state of all inputs
     */
    getState() {
        let state = {"images": this.images.getState()},
            controllerArray = this.getStatefulControllers();
        for (let controller of controllerArray) {
            state = {...state, ...controller.getState()};
        }
        return state;
    }

    /**
     * Determine if the state has anything worthwhile to save.
     */
    shouldSaveState() {
        let state = this.getState();
        if (!isEmpty(state.prompts)) {
            if (!isEmpty(state.prompts.prompt) || !isEmpty(state.prompts.negativePrompt)) {
                return true;
            }
        }
        return !isEmpty(state.images);
    }

    /**
     * Sets the current state of all inputs
     */
    async setState(newState, saveHistory = false) {
        if (saveHistory === true && this.shouldSaveState()) {
            await this.autosave(false);
            await this.history.flush(newState);
        }
        let controllerArray = this.getStatefulControllers();
        for (let controller of controllerArray) {
            await controller.setState(newState);
        }
        if (!isEmpty(newState.canvas)) {
            if (!isEmpty(newState.canvas.width)) this.images.width = newState.canvas.width;
            if (!isEmpty(newState.canvas.height)) this.images.height = newState.canvas.height;
        }
        if (newState.images !== undefined && newState.images !== null) {
            this.engine.hideSampleChooser();
            this.images.hideCurrentInvocation();
            this.images.setState(newState.images);
        }
    }

    /**
     * Resets state to default values
     */
    async resetState(saveHistory = true) {
        let state = {"images": []},
            controllerArray = this.getStatefulControllers();
        for (let controller of controllerArray) {
            state = {...state, ...controller.getDefaultState()};
        }
        await this.setState(state, saveHistory);
    }

    /**
     * Initializes state from an image
     */
    async initializeStateFromImage(image, saveHistory = true) {
        try {
            let baseState = {},
                keepState = await this.yesNo("Would you like to keep settings?<br /><br />This will maintain things like prompts and other global settings the same while only changing the dimensions to match the image."),
                controllerArray = this.getStatefulControllers();
            for (let controller of controllerArray) {
                if (keepState) {
                    baseState = {...baseState, ...controller.getState()};
                } else {
                    baseState = {...baseState, ...controller.getDefaultState()};
                }
            }
            if (isEmpty(baseState.canvas)) {
                baseState.canvas = {};
            }
            baseState.canvas.width = image.width;
            baseState.canvas.height = image.height;
            baseState.images = [ImageEditorView.getNodeDataForImage(image)];

            this.engine.hideSampleChooser();
            this.images.hideCurrentInvocation();
            this.images.width = image.width;
            this.images.height = image.height;

            await this.setState(baseState, saveHistory);
        } catch(e) {
            // pass
        }
    }

    /**
     * The global onKeyPress fires keyboard shortcuts.
     */
    onKeyPress(e) {
        // TODO
    }

    /**
     * The handler called when the popState event is called.
     * @param Event $e The popState event
     */
    async popState(e) {
        // TODO
    }
}

export { Application };
