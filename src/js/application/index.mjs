/** @module application/index */
import {
    isEmpty,
    getQueryParameters,
    getDataParameters,
    waitFor,
    createEvent,
    merge,
    sleep
} from "../base/helpers.mjs";
import { Session } from "../base/session.mjs";
import { Publisher } from "../base/publisher.mjs";
import { TooltipHelper } from "../common/tooltip.mjs";
import { MenuView, SidebarView } from "../view/menu.mjs";
import { StatusView } from "../view/status.mjs";
import { NotificationCenterView } from "../view/notifications.mjs";
import { WindowsView } from "../nodes/windows.mjs";
import { ImageView, BackgroundImageView } from "../view/image.mjs";
import { Model } from "../model/enfugue.mjs";
import { View } from "../view/base.mjs";
import { ControlsHelperView } from "../view/controls.mjs";
import { FileNameFormView } from "../forms/enfugue/files.mjs";
import { StringInputView } from "../forms/input.mjs";
import { InvocationController } from "../controller/common/invocation.mjs";
import { SamplesController } from "../controller/common/samples.mjs";
import { ModelPickerController } from "../controller/common/model-picker.mjs";
import { ModelManagerController } from "../controller/common/model-manager.mjs";
import { DownloadsController } from "../controller/common/downloads.mjs";
import { LayersController } from "../controller/common/layers.mjs";
import { PromptTravelController } from "../controller/common/prompts.mjs";
import { AnimationsController } from "../controller/common/animations.mjs";
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
     * @var object keyboard shortcuts for menu categories
     */
    static menuCategoryShortcuts = {
        "file": "f",
        "models": "m",
        "system": "s",
        "help": "h"
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
        if (!isEmpty(window.enfugue) && !isEmpty(window.enfugue.config)) {
            config = merge(config, window.enfugue.config);
        }
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
        this.container.classList.add("loader");
        this.container.classList.add("loading");
        this.session = Session.getScope("enfugue", 60 * 60 * 1000 * 24 * 30); // 1 month session
        this.model = new Model(this.config);
        this.menu = new MenuView(this.config);
        this.sidebar = new SidebarView(this.config);
        this.windows = new WindowsView(this.config);
        this.notifications = new NotificationCenterView(this.config);
        this.history = new HistoryDatabase(this.config.history.size, this.config.debug);
        this.images = new ImageEditorView(this);
        this.controlsHelper = new ControlsHelperView(this.config);

        this.container.appendChild(await this.menu.render());
        this.container.appendChild(await this.sidebar.render());
        this.container.appendChild(await this.images.render());
        this.container.appendChild(await this.windows.render());
        this.container.appendChild(await this.notifications.render());
        this.container.appendChild(await this.controlsHelper.render());

        if (this.config.debug) console.log("Starting animations.");
        await this.startAnimations();
        if (this.config.debug) console.log("Registering dynamic inputs.");
        await this.registerDynamicInputs();
        if (this.config.debug) console.log("Registering download controllers.");
        await this.registerDownloadsControllers();
        if (this.config.debug) console.log("Registering animation controllers.");
        await this.registerAnimationsControllers();
        if (this.config.debug) console.log("Registering model controllers.");
        await this.registerModelControllers();
        if (this.config.debug) console.log("Registering invocation controllers.");
        await this.registerInvocationControllers();
        if (this.config.debug) console.log("Registering sample controllers.");
        await this.registerSampleControllers();
        if (this.config.debug) console.log("Registering layer controllers.");
        await this.registerLayersControllers();
        if (this.config.debug) console.log("Registering prompt controllers.");
        await this.registerPromptControllers();
        if (this.config.debug) console.log("Registering menu controllers.");
        await this.registerMenuControllers();
        if (this.config.debug) console.log("Registering sidebar controllers.");
        await this.registerSidebarControllers();
        if (this.config.debug) console.log("Starting autosave.");
        await this.startAutosave();
        if (this.config.debug) console.log("Starting announcement check.");
        await this.startAnnouncements();
        if (this.config.debug) console.log("Starting keepalive.");
        await this.startKeepalive();
        if (this.config.debug) console.log("Registering authentication.");
        await this.registerLogout();

        window.onpopstate = (e) => this.popState(e);
        document.addEventListener("dragover", (e) => this.onDragOver(e));
        document.addEventListener("drop", (e) => this.onDrop(e));
        document.addEventListener("paste", (e) => this.onPaste(e));
        document.addEventListener("keypress", (e) => this.onKeyPress(e));
        document.addEventListener("keyup", (e) => this.onKeyUp(e));
        document.addEventListener("keydown", (e) => this.onKeyDown(e));

        if (this.config.debug) console.log("Application initialization complete.");

        this.publish("applicationReady");
        this.container.classList.remove("loading");
    }

    /**
     * Starts the announcements controller which will get necessary initialization
     * actions as well as versions from remote.
     */
    async startAnnouncements() {
        if (this.userIsAdmin && !this.userIsSandboxed) {
            this.announcements = new AnnouncementsController(this);
            await this.announcements.initialize();
        }
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
        if (!this.userIsAdmin) {
            // Remove other input
            delete DefaultVaeInputView.defaultOptions.other;
        }
        CheckpointInputView.defaultOptions = async () => {
            let checkpoints = await this.model.get("/checkpoints");
            return checkpoints.reduce((carry, datum) => {
                if (!isEmpty(datum.directory) && datum.directory !== ".") {
                    carry[datum.name] = `<strong>${datum.name}</strong><span class='note' style='margin-left: 2px'>(${datum.directory})</note>`;
                } else {
                    carry[datum.name] = datum.name;
                }
                return carry;
            }, {});
        };
        LoraInputView.defaultOptions = async () => {
            let models = await this.model.get("/lora");
            return models.reduce((carry, datum) => {
                if (!isEmpty(datum.directory) && datum.directory !== ".") {
                    carry[datum.name] = `<strong>${datum.name}</strong><span class='note' style='margin-left: 2px'>(${datum.directory})</note>`;
                } else {
                    carry[datum.name] = datum.name;
                }
                return carry;
            }, {});
        };
        LycorisInputView.defaultOptions = async () => {
            let models = await this.model.get("/lycoris");
            return models.reduce((carry, datum) => {
                if (!isEmpty(datum.directory) && datum.directory !== ".") {
                    carry[datum.name] = `<strong>${datum.name}</strong><span class='note' style='margin-left: 2px'>(${datum.directory})</note>`;
                } else {
                    carry[datum.name] = datum.name;
                }
                return carry;
            }, {});
        };
        InversionInputView.defaultOptions = async () => {
            let models = await this.model.get("/inversions");
            return models.reduce((carry, datum) => {
                if (!isEmpty(datum.directory) && datum.directory !== ".") {
                    carry[datum.name] = `<strong>${datum.name}</strong><span class='note' style='margin-left: 2px'>(${datum.directory})</note>`;
                } else {
                    carry[datum.name] = datum.name;
                }
                return carry;
            }, {});
        };
        ModelPickerInputView.defaultOptions = async () => {
            let allModels = await this.model.get("/model-options");
            return allModels.reduce((carry, datum) => {
                let typeString = isEmpty(datum.type)
                    ? ""
                    :datum.type === "checkpoint"
                        ? "Checkpoint"
                        : datum.type === "checkpoint+diffusers"
                            ? "Checkpoint + Diffusers Cache"
                            : datum.type === "diffusers"
                                ? "Diffusers Cache"
                                : "Preconfigured Model";
                if (!isEmpty(datum.directory) && datum.directory !== ".") {
                    carry[`${datum.type}/${datum.name}`] = `<strong>${datum.name}</strong><span class='note' style='margin-left: 2px'>(${datum.directory})</note></span><em>${typeString}</em>`;
                } else {
                    carry[`${datum.type}/${datum.name}`] = `<strong>${datum.name}</strong><em>${typeString}</em>`;
                }
                return carry;
            }, {});
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
     * Creates the samples manager.
     */
    async registerSampleControllers() {
        this.samples = new SamplesController(this);
        await this.samples.initialize();
    }

    /**
     * Creates the layers manager (handles multiple images)
     */
    async registerLayersControllers() {
        this.layers = new LayersController(this);
        await this.layers.initialize();
    }

    /**
     * Creates the prompts manager (handles prompt travel)
     */
    async registerPromptControllers() {
        this.prompts = new PromptTravelController(this);
        await this.prompts.initialize();
    }

    /**
     * Creates the animation manager (enable/disable animations.)
     */
    async registerAnimationsControllers() {
        this.animation = new AnimationsController(this);
        await this.animation.initialize();
    }

    /**
     * Returns true if the frontend thinks this user is an admin
     * All API calls are authenticated
     * 
     * @return bool True if the user is admin
     */
    get userIsAdmin() {
        return !isEmpty(window.enfugue) && window.enfugue.admin === true;
    }

    /**
     * Returns true if the frontend thinks this user is sandboxed
     * 
     * @return bool True if the user is admin
     */
    get userIsSandboxed() {
        return !isEmpty(window.enfugue) && window.enfugue.sandboxed === true;
    }

    /**
     * Returns the menu categories to import based on user context
     */
    getMenuCategories() {
        let menuCategories = {...this.constructor.menuCategories};
        if (this.userIsAdmin) {
            menuCategories = {...menuCategories, ...this.constructor.adminMenuCategories };
            if (this.userIsSandboxed) {
                delete menuCategories.system;
            }
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
            let menuCategoryShortcut = this.constructor.menuCategoryShortcuts[menuCategoryName];
            this.menuControllers[menuCategoryName] = [];
            try {
                let menuCategoryModule = await import(`../controller/${menuCategoryName}/index.autogenerated.mjs`);
                let menuCategory = await this.menu.addCategory(menuCategoryLabel, menuCategoryShortcut);
                for (let menuControllerPath of menuCategoryModule.Index) {
                    try {
                        let menuItemControllerModule = await import(`../controller/${menuCategoryName}/${menuControllerPath}`);
                        let menuItemControllerClass = menuItemControllerModule.MenuController;
                        if (isEmpty(menuItemControllerClass)) {
                            throw "Module does not provide a 'MenuController' export.";
                        }
                        if (!menuItemControllerClass.isDisabled()) {
                            let menuItem = await menuCategory.addItem(menuItemControllerClass.menuName, menuItemControllerClass.menuIcon, menuItemControllerClass.menuShortcut);
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
                if (this.config.debug) {
                    console.log("Loading autosaved state", existingAutosave);
                }
                await this.setState(existingAutosave);
                this.notifications.push("info", "Session Restored", "Your last autosaved session was successfully loaded.");
                if (!isEmpty(this.images.node)) {
                    let reset = this.images.node.find("enfugue-node-editor-zoom-reset");
                    if (!isEmpty(reset)) {
                        reset.trigger("click");
                    }
                }
            }
            const autosaveInterval = this.config.model.autosave.interval || 30000;
            setInterval(() => this.autosave(), autosaveInterval);
        } catch(e) {
            console.error(e);
            this.notifications.push("warn", "History Disabled", "Couldn't open IndexedDB, history and autosave are disabled.");
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
                this.loadFile(item.getAsFile());
            } else {
                item.getAsString((text) => this.onTextPaste(text));
            }
        }
    }

    /**
     * Checks loaded image metadata
     */
    getStateFromMetadata(metadata) {
        if (!isEmpty(metadata.EnfugueUIState)) {
            return JSON.parse(metadata.EnfugueUIState);
        }
        return {};
    }

    /**
     * This handler reads the file passed and determines what it is,
     * then loads it onto the canvas if possible.
     */
    async loadFile(file, name = "Image") {
        let reader = new FileReader();
        reader.addEventListener("load", async () => {
            let fileType = reader.result.substring(5, reader.result.indexOf(";")),
                contentStart = fileType.length + 13,
                contentAsText = () => atob(reader.result.substring(contentStart)),
                imageView;

            switch (fileType) {
                case "application/json":
                    await this.setState(JSON.parse(contentAsText()));
                    this.notifications.push("info", "Generation Settings Loaded", "Image generation settings were successfully retrieved from image metadata.");
                    break;
                    break;
                case "image/png":
                    imageView = new BackgroundImageView(this.config, reader.result);
                    await imageView.waitForLoad();
                    let stateData = this.getStateFromMetadata(imageView.metadata);
                    if (!isEmpty(stateData)) {
                        if (await this.yesNo("It looks like this image was made with Enfugue. Would you like to load the identified generation settings?")) {
                            await this.setState(stateData);
                            this.notifications.push("info", "Generation Settings Loaded", "Image generation settings were successfully retrieved from image metadata.");
                            return;
                        }
                    }
                case "image/gif":
                case "image/avif":
                case "image/jpeg":
                case "image/bmp":
                case "image/tiff":
                case "image/x-icon":
                case "image/webp":
                    if (isEmpty(imageView)) {
                        imageView = new BackgroundImageView(this.config, reader.result, false);
                    }
                    this.samples.showCanvas();
                    this.layers.addImageLayer(imageView);
                    break;
                case "video/mp4":
                    this.samples.showCanvas();
                    this.layers.addVideoLayer(reader.result);
                    break;
                default:
                    this.notifications.push("warn", "Unhandled File Type", `File type "${fileType}" is not handled by Enfugue.`);
                    break;
            }
        });
        reader.readAsDataURL(file);
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
        let controllerArray = [
            this.modelPicker,
            this.layers,
            this.samples,
            this.prompts,
        ].concat(this.sidebarControllers);
        for (let controllerName in this.menuControllers) {
            controllerArray = controllerArray.concat(this.menuControllers[controllerName]);
        }
        return controllerArray;
    }

    /**
     * Gets current state of all inputs
     */
    getState(includeImages = true) {
        let state = {},
            controllerArray = this.getStatefulControllers();
        for (let controller of controllerArray) {
            state = {...state, ...controller.getState(includeImages)};
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
        return !isEmpty(state.layers);
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
        if (!isEmpty(newState.canvas)) {
            this.images.setDimension(newState.canvas.width, newState.canvas.height);
        }
        for (let controller of controllerArray) {
            await controller.setState(newState);
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
    async initializeStateFromImage(image, saveHistory = true, keepState = null, overrideState = null) {
        try {
            let baseState = {},
                controllerArray = this.getStatefulControllers();

            if (keepState === null) {
                keepState = await this.yesNo("Would you like to keep settings?<br /><br />This will maintain things like prompts and other global settings the same while only changing the dimensions to match the image.");
            }

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
            baseState.images = {
                nodes: [ImageEditorView.getNodeDataForImage(image)],
                image: null
            };

            if (!isEmpty(overrideState)) {
                for (let overrideKey in overrideState) {
                    if (baseState[overrideKey] !== undefined) {
                        if (overrideState[overrideKey] === null) {
                            baseState[overrideKey] = null;
                        } else if (typeof baseState[overrideKey] == "object" && typeof overrideState[overrideKey] == "object") {
                            baseState[overrideKey] = {...baseState[overrideKey], ...overrideState[overrideKey]};
                        } else {
                            baseState[overrideKey] = overrideState[overrideKey];
                        }
                    }
                }
            }
            this.images.hideCurrentInvocation();
            this.engine.hideSampleChooser();
            await sleep(1); // Sleep 1 frame
            await this.setState(baseState, saveHistory);
        } catch(e) {
            console.error(e);
            // pass
        }
    }

    /**
     * Passes through to invoke, setting state in the process
     */
    invoke(kwargs) {
        kwargs.state = this.getState(false);
        return this.engine.invoke(kwargs);
    }

    /**
     * The global onKeyPress fires keyboard shortcuts.
     */
    onKeyPress(e) {
        if (e.shiftKey) {
            this.menu.fireCategoryShortcut(e.key);
            this.publish("keyboardShortcut", e.key);
        }
    }

    /**
     * The global keydown event highlights menu shortcuts.
     */
    onKeyDown(e){
        if (e.key === "Shift") {
            this.menu.addClass("highlight");
        }
    }

    /**
     * The global keyup event stops highlighting menu shortcuts.
     */
    onKeyUp(e){
        if (e.key === "Shift") {
            this.menu.removeClass("highlight");
        }
    }

    /**
     * Don't do anything on drag over.
     */
    onDragOver(e){
        e.preventDefault();
    }

    /**
     * On drop, treat as file paste.
     */
    onDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        try {
            this.loadFile(e.dataTransfer.files[0]);
        } catch(e) {
            console.warn(e);
        }
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
