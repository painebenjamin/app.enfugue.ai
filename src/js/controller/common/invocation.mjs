import { ImageInspectorView } from "../../view/image.mjs";
import { isEmpty, isEquivalent, waitFor, humanDuration } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";

const E = new ElementBuilder({
    "invocationLoading": "enfugue-invocation-loading",
    "invocationLoaded": "enfugue-invocation-loaded",
    "invocationDuration": "enfugue-invocation-duration",
    "invocationIterations": "enfugue-invocation-iterations",
    "invocationRemaining": "enfugue-invocation-remaining",
    "invocationSampleChooser": "enfugue-invocation-sample-chooser",
    "invocationSample": "enfugue-invocation-sample",
    "invocationStop": "enfugue-invocation-stop"
});

/**
 * This class manages invocations of the enfugue engine.
 */
class InvocationController extends Controller {
    /**
     * @var int The number of characters after which to truncate titles.
     */
    static truncatePromptLength = 36;

    /**
     * @param Application application The overall application state container.
     */
    constructor(application) {
        super(application);
        this.kwargs = {};
    }

    /**
     * @return int Either the configured width or default width
     */
    get width() {
        return this.kwargs.width || this.application.config.model.invocation.width;
    }
    
    /**
     * @param int newWidth The new width to invoke with.
     */
    set width(newWidth) {
        if (this.width !== newWidth){
            this.publish("engineWidthChange", newWidth);
        }
        this.kwargs.width = newWidth;
    }
    
    /**
     * @return int Either the configured height or default height
     */
    get height() {
        return this.kwargs.height || this.application.config.model.invocation.height;
    }

    /**
     * @param int newHeight The new height to invoke with.
     */
    set height(newHeight) {
        if (this.height !== newHeight){
            this.publish("engineHeightChange", newHeight);
        }
        this.kwargs.height = newHeight;
    }

    /**
     * @return The chunking size; i.e. how many pixels the rendering window moves by during multidiffusion.
     */
    get chunkingSize() {
        return this.kwargs.chunking_size || this.application.config.model.invocation.chunkingSize;
    }

    /**
     * @param int Sets the new chunking size. 0 disables multidiffusion.
     */
    set chunkingSize(newSize) {
        if (this.chunkingSize !== newSize){
            this.publish("engineChunkingSizeChange", newSize);
        }
        this.kwargs.chunking_size = newSize
    }
    
    /**
     * @return The chunking blur; i.e. how many pixels the rendering window feathers by during multidiffusion.
     */
    get chunkingBlur() {
        return this.kwargs.chunking_blur || this.application.config.model.invocation.chunkingBlur;
    }

    /**
     * @param int Sets the new chunking blur. 0 disables multidiffusion.
     */
    set chunkingBlur(newBlur) {
        if (this.chunkingBlur !== newBlur){
            this.publish("engineChunkingBlurChange", newBlur);
        }
        this.kwargs.chunking_blur = newBlur
    }

    /**
     * @return string Either the configured prompt or empty string.
     */
    get prompt() {
        return this.kwargs.prompt || "";
    }

    /**
     * @param string newPrompt Sets the prompt to invoke with.
     */
    set prompt(newPrompt) {
        if (this.prompt !== newPrompt) {
            this.publish("enginePromptChange", newPrompt);
        }
        this.kwargs.prompt = newPrompt;
    }

    /**
     * @return string Either the configured negative prompt or empty string.
     */
    get negativePrompt() {
        return this.kwargs.negative_prompt || "";
    }

    /**
     * @param string newPrompt Sets the negative prompt to invoke with.
     */
    set negativePrompt(newPrompt) {
        if (this.negativePrompt !== newPrompt) {
            this.publish("engineNegativePromptChange", newPrompt);
        }
        this.kwargs.negative_prompt = newPrompt;
    }

    /**
     * @return int The numbers of samples to generate at the same time.
     */
    get samples() {
        return this.kwargs.samples || 1;
    }

    /**
     * @param int newSamples The new number of samples to generate.
     */
    set samples(newSamples) {
        if (this.samples !== newSamples) {
            this.publish("engineSamplesChange", newSamples);
        }
            
        this.kwargs.samples = newSamples;
    }

    /**
     * @return ?int Either the set seed, or empty (random generation)
     */
    get seed() {
        return this.kwargs.seed;
    }

    /**
     * @param ?int The new seed to set, or null for random.
     */
    set seed(newSeed) {
        if (this.seed !== newSeed) {
            this.publish("engineSeedChange", newSeed);
        }
        this.kwargs.seed = newSeed;
    }

    /**
     * @return float The guidance scale, i.e. how closely prompts are followed.
     */
    get guidanceScale() {
        return this.kwargs.guidance_scale || this.application.config.model.invocation.guidanceScale;
    }

    /**
     * @param float newGuidanceScale Sets a new guidance scale.
     */
    set guidanceScale(newGuidanceScale) {
        if (this.guidanceScale !== newGuidanceScale) {
            this.publish("engineGuidanceScaleChange", newGuidanceScale);
        }
        this.kwargs.guidance_scale = newGuidanceScale;
    }

    /**
     * @return int The number of denoising steps.
     */
    get inferenceSteps() {
        return this.kwargs.num_inference_steps || this.application.config.model.invocation.inferenceSteps;
    }

    /**
     * @param int Sets the new number of denoising steps.
     */
    set inferenceSteps(newInferenceSteps) {
        if (this.inferenceSteps !== newInferenceSteps) {
            this.publish("engineInferenceStepsChange", newInferenceSteps);
        }
        this.kwargs.num_inference_steps = newInferenceSteps;
    }

    /**
     * @return ?string The model to use, null for default.
     */
    get model() {
        return this.kwargs.model;
    }

    /**
     * @param ?string The new model to use, or null for default.
     */
    set model(newModel) {
        if (this.model !== newModel) {
            this.publish("engineModelChange", newModel);
        }
        this.kwargs.model = newModel;
    }

    /**
     * @return int The output scale (1 by default.)
     */
    get outscale() {
        return this.kwargs.outscale || 1;
    }

    /**
     * @param int newOutscale The new outscale value.
     */
    set outscale(newOutscale) {
        if (this.outscale !== newOutscale) {
            this.publish("engineOutscaleChange", newOutscale);
        }
        this.kwargs.outscale = parseInt(newOutscale);
    }

    /**
     * @return ?str|array<str> The upscale method(s), null by default
     */
    get upscale() {
        return this.kwargs.upscale;
    }

    /**
     * @param str|array<str> The new upscale method(s)
     */
    set upscale(newUpscale) {
        if (!isEquivalent(this.upscale, newUpscale)) {
            this.publish("engineUpscaleChange", newUpscale);
        }
        this.kwargs.upscale = newUpscale;
    }

    /**
     * @return bool Whether or not iterative upscaling is enabled. Default false.
     */
    get upscaleIterative() {
        return this.kwargs.upscale_iterative === true;
    }

    /**
     * @param bool The new value for iterative upscaling.
     */
    set upscaleIterative(newUpscaleIterative) {
        if (this.upscaleIterative !== newUpscaleIterative) {
            this.publish("engineUpscaleIterativeChange", newUpscaleIterative);
        }
        this.kwargs.upscale_iterative = newUpscaleIterative;
    }
    
    /**
     * @return bool Whether or not diffusion upscaling is enabled. Default false.
     */
    get upscaleDiffusion() {
        return this.kwargs.upscale_diffusion === true;
    }

    /**
     * @param bool The new value for diffusion upscaling.
     */
    set upscaleDiffusion(newUpscaleDiffusion) {
        if (this.upscaleDiffusion !== newUpscaleDiffusion) {
            this.publish("engineUpscaleDiffusionChange", newUpscaleDiffusion);
        }
        this.kwargs.upscale_diffusion = newUpscaleDiffusion;
    }

    /**
     * @return ?str|array<str> The prompt(s) to use during upscale diffusion. Default null.
     */
    get upscaleDiffusionPrompt() {
        return this.kwargs.upscale_diffusion_prompt;
    }

    /**
     * @param str|array<str> The new prompt(s) to use during upscale diffusion.
     */
    set upscaleDiffusionPrompt(newPrompt) {
        if (!isEquivalent(this.upscaleDiffusionPrompt, newPrompt)) {
            this.publish("engineUpscaleDiffusionPromptChange", newPrompt);
        }
        this.kwargs.upscale_diffusion_prompt = newPrompt;
    }
   
    /**
     * @return ?str|array<str> The prompt(s) to use during upscale diffusion. Default null.
     */
    get upscaleDiffusionNegativePrompt() {
        return this.kwargs.upscale_diffusion_negative_prompt;
    }

    /**
     * @param str|array<str> The new prompt(s) to use during upscale diffusion.
     */
    set upscaleDiffusionNegativePrompt(newNegativePrompt) {
        if (!isEquivalent(this.upscaleDiffusionNegativePrompt, newNegativePrompt)) {
            this.publish("engineUpscaleDiffusionNegativePromptChange", newNegativePrompt);
        }
        this.kwargs.upscale_diffusion_negative_prompt = newNegativePrompt;
    }

    /**
     * @return ?int|array<int> The steps to use during upscale diffusion. Default 100.
     */
    get upscaleDiffusionSteps() {
        return this.kwargs.upscale_diffusion_steps || this.application.config.model.invocation.upscaleDiffusionSteps;
    }

    /**
     * @param int|array<int> The new steps to use during upscale diffusion.
     */
    set upscaleDiffusionSteps(newSteps) {
        if (!isEquivalent(this.upscaleDiffusionSteps, newSteps)) {
            this.publish("engineUpscaleDiffusionStepsChange", newSteps);
        }
        this.kwargs.upscale_diffusion_steps = newSteps;
    }
    
    /**
     * @return ?int|array<int> The strength(s) to use during upscale diffusion.
     */
    get upscaleDiffusionStrength() {
        return this.kwargs.upscale_diffusion_strength || this.application.config.model.invocation.upscaleDiffusionStrength;
    }

    /**
     * @param float|array<float> The new strength(s) to use during upscale diffusion.
     */
    set upscaleDiffusionStrength(newStrength) {
        if (!isEquivalent(this.upscaleDiffusionStrength, newStrength)) {
            this.publish("engineUpscaleDiffusionStrengthChange", newStrength);
        }
        this.kwargs.upscale_diffusion_strength = newStrength;
    }
    
    /**
     * @return ?float|array<float> The guidance scale(s) to use during upscale diffusion.
     */
    get upscaleDiffusionGuidanceScale() {
        return this.kwargs.upscale_diffusion_guidance_scale || this.application.config.model.invocation.upscaleDiffusionGuidanceScale;
    }

    /**
     * @param float|array<float> The new guidance scale(s) to use during upscale diffusion.
     */
    set upscaleDiffusionGuidanceScale(newGuidanceScale) {
        if (!isEquivalent(this.upscaleDiffusionGuidanceScale, newGuidanceScale)) {
            this.publish("engineUpscaleDiffusionGuidanceScaleChange", newGuidanceScale);
        }
        this.kwargs.upscale_diffusion_guidance_scale = newGuidanceScale;
    }
    
    /**
     * @return int The chunking size to use during upscale diffusion
     */
    get upscaleDiffusionChunkingSize() {
        return this.kwargs.upscale_diffusion_chunking_size || this.application.config.model.invocation.upscaleDiffusionChunkingSize;
    }

    /**i
     * @param int The new chunking size to use during upscale diffusion.
     */
    set upscaleDiffusionChunkingSize(newChunkingSize) {
        if (this.upscaleDiffusionChunkingSize !== newChunkingSize) {
            this.publish("engineUpscaleDiffusionChunkingSizeChange", newChunkingSize);
        }
        this.kwargs.upscale_diffusion_chunking_size = newChunkingSize;
    }
    
    /**
     * @return int The chunking blur to use during upscale diffusion
     */
    get upscaleDiffusionChunkingBlur() {
        return this.kwargs.upscale_diffusion_chunking_blur || this.application.config.model.invocation.upscaleDiffusionChunkingBlur;
    }

    /**
     * @param int The new chunking blur to use during upscale diffusion.
     */
    set upscaleDiffusionChunkingBlur(newChunkingBlur) {
        if (this.upscaleDiffusionChunkingBlur !== newChunkingBlur) {
            this.publish("engineUpscaleDiffusionChunkingBlurChange", newChunkingBlur);
        }
        this.kwargs.upscale_diffusion_chunking_blur = newChunkingBlur;
    }
    
    /**
     * @return bool Whether or not to scale the size of the chunk with each upscale iteration. Default true.
     */
    get upscaleDiffusionScaleChunkingSize() {
        return this.kwargs.upscale_diffusion_scale_chunking_size !== false;
    }

    /**
     * @param bool The new value of the flag.
     */
    set upscaleDiffusionScaleChunkingSize(newScaleChunkingSize) {
        if (this.upscaleDiffusionScaleChunkingSize !== newScaleChunkingSize) {
            this.publish("engineUpscaleDiffusionScaleChunkingSizeChange", newScaleChunkingSize);
        }
        this.kwargs.upscale_diffusion_scale_chunking_size = newScaleChunkingSize;
    }
    
    /**
     * @return bool Whether or not to scale the size of the chunk with each upscale iteration. Default true.
     */
    get upscaleDiffusionScaleChunkingBlur() {
        return this.kwargs.upscale_diffusion_scale_chunking_blur !== false;
    }

    /**
     * @param bool The new value of the flag.
     */
    set upscaleDiffusionScaleChunkingBlur(newScaleChunkingBlur) {
        if (this.upscaleDiffusionScaleChunkingBlur !== newScaleChunkingBlur) {
            this.publish("engineUpscaleDiffusionScaleChunkingBlurChange", newScaleChunkingBlur);
        }
        this.kwargs.upscale_diffusion_scale_chunking_blur = newScaleChunkingBlur;
    }
   
    /**
     * @return array<string> Any number of controlnets to use during upscaling. Default null (none)
     */
    get upscaleDiffusionControlnet() {
        return this.kwargs.upscale_diffusion_controlnet || null;
    }

    /**
     * @param bool The new value of the flag.
     */
    set upscaleDiffusionControlnet(newControlnet) {
        if (!isEquivalent(this.upscaleDiffusionControlnet, newControlnet)) {
            this.publish("engineUpscaleDiffusionControlnetChange", newControlnet);
        }
        this.kwargs.upscale_diffusion_controlnet = newControlnet;
    }

    /**
     * On initialization, create DOM elements related to invocations.
     */
    async initialize() {
        this.loadingBar = E.invocationLoading().content(
            E.invocationLoaded().addClass("sliding-gradient"),
            E.invocationDuration(),
            E.invocationIterations(),
            E.invocationRemaining().hide()
        );
        this.invocationSampleChooser = E.invocationSampleChooser().hide();
        this.invocationStop = E.invocationStop().content("Stop").on("click", () => { this.stopInvocation() });
        (await this.images.getNode()).append(this.loadingBar).append(this.invocationSampleChooser).append(this.invocationStop);
    }

    /**
     * Hides the sample chooser from outside the controller.
     */
    hideSampleChooser() {
        this.invocationSampleChooser.hide();
    }

    /**
     * Starts an invocation.
     *
     * @param object payload    The keyword arguments to send to the server.
     * @param bool   detached   Whether or not to invoke in a standalone window. Default false.
     */
    async invoke(payload, detached = false) {
        payload = isEmpty(payload) ? {} : payload;
        let invocationPayload = {...this.kwargs, ...payload};
        if (isEmpty(invocationPayload.prompt)) {
            this.notify("Error", "Missing Prompt", "A prompt is required.");
            return;
        }
    
        let result = await this.application.model.post(
            "invoke", 
            null, 
            null, 
            invocationPayload
        );

        this.invocationStop.addClass("ready");
        if (!isEmpty(result.uuid)) {
            if (detached) {
                let invocationName = invocationPayload.prompt.substring(0, this.constructor.truncatePromptLength);
                if (invocationName.length === this.constructor.truncatePromptLength) {
                    invocationName += "...";
                }
                await this.startDetachedInvocationMonitor(
                    invocationName,
                    result.uuid,
                    parseInt(invocationPayload.width) || 512,
                    parseInt(invocationPayload.height) || 512
                );
            } else {
                await this.canvasInvocation(result.uuid);
                this.invocationStop.removeClass("ready");
            }
        }
    }

    /**
     * Stops the engine.
     */
    async stopInvocation() {
        if (!this.invocationStop.hasClass("ready")) return;
        try {
            await this.application.model.post("/invocation/stop");
            this.invocationStop.removeClass("ready");
            this.notify("info", "Stopped", "Successfully stopped engine.");
        } catch(e) {
            let errorMessage = `${e}`;
            if (!isEmpty(e.detail)) errorMessage = e.detail;
            else if (!isEmpty(e.title)) errorMessage = e.title;
            this.notify("error", "Error", `Received an error when stopping. The engine may still be stopped, wait a moment and check again. ${errorMessage}`);
        }
    }

    /**
     * This is the meat and potatoes of watching an invocation as it goes; this method will be called by implementing functions with callbacks.
     * We estimate using total duration, this will end up being more accurate over the entirety of the invocation is they will typically
     * start slow, speed up, then slow down again.
     *
     * @param string uuid The UUID of the invocation.
     * @param callable onImagesReceived A callback that will receive (list<str> $images, bool $complete) when images are retrieved.
     * @param callable onError A callback that is called when an error occur.
     * @param callable onEstimatedDuration A callback that will receive (int $millisecondsRemaining) when new estimates are available.
     */
    async monitorInvocation(uuid, onImagesReceived, onError, onEstimatedDuration) {
        const initialInterval = this.application.config.model.invocation.interval || 1000;
        const queuedInterval = this.application.config.model.queue.interval || 5000;
        const consecutiveErrorCutoff = this.application.config.model.invocation.errors.consecutive || 2;

        if (onImagesReceived === undefined) onImagesReceived = () => {};
        if (onError === undefined) onError = () => {};
        if (onEstimatedDuration === undefined) onEstimatedDuration = () => {};

        let start = (new Date()).getTime(),
            lastStep,
            lastTotal,
            lastRate,
            lastDuration,
            lastStepDeltaTime,
            lastTotalDeltaTime = start,
            getEstimatedDurationRemaining = () => {
                let averageStepsPerMillisecond = lastStep/(lastStepDeltaTime-lastTotalDeltaTime),
                    currentStepsPerMillisecond = isEmpty(lastRate) ? averageStepsPerMillisecond : lastRate / 1000,
                    weightedStepsPerMillisecond = (currentStepsPerMillisecond * 0.75) + (averageStepsPerMillisecond * 0.25),
                    millisecondsRemainingAtDelta = (lastTotal-lastStep)/weightedStepsPerMillisecond,
                    millisecondsRemainingNow = millisecondsRemainingAtDelta-((new Date()).getTime()-lastStepDeltaTime);

                if (isNaN(millisecondsRemainingNow)) {
                    millisecondsRemainingNow = Infinity;
                }
                
                onEstimatedDuration(millisecondsRemainingNow, lastStep, lastTotal, lastRate, lastDuration);
                return millisecondsRemainingNow;
            },
            getInterval = (invokeResult) => {
                if (invokeResult.status === "queued") {
                    return queuedInterval;
                }
                return Math.min(Math.max(initialInterval, getEstimatedDurationRemaining() / 2), queuedInterval);
            },
            checkInvocationTimer,
            checkInvocation = async () => {
                let invokeResult, lastError;
                for (let i = 0; i < consecutiveErrorCutoff; i++) {
                    try {
                        invokeResult = await this.application.model.get(`/invocation/${uuid}`);
                    } catch(e) {
                        lastError = e;
                    }
                }
                if (isEmpty(invokeResult)) {
                    this.notify("error", "Invocation Error", `${consecutiveErrorCutoff} consecutive errors were detected communicating with the server. Please check the server logs. The last error was: ${lastError}`);
                    onError();
                    return;
                }

                if (invokeResult.total !== lastTotal) {
                    if (!isEmpty(lastTotal)) {
                        lastTotalDeltaTime = (new Date()).getTime();
                    }
                    lastTotal = invokeResult.total;
                }
                if (invokeResult.step !== lastStep) {
                    lastStep = invokeResult.step;
                    lastStepDeltaTime = (new Date()).getTime();
                }
                if (invokeResult.rate !== lastRate) {
                    lastRate = invokeResult.rate;
                }
                lastDuration = invokeResult.duration;
                if (!isEmpty(invokeResult.images)) {
                    let imagePaths = invokeResult.images.map((imageName) => `/api/invocation/${imageName}`),
                        isCompleted = invokeResult.status === "completed";
                    onImagesReceived(imagePaths, isCompleted);
                }
                if (invokeResult.status === "error") {
                    this.notify("error", "Invocation Failed", invokeResult.message);
                    onError();
                    return;
                } else if (invokeResult.status !== "completed") {
                    checkInvocationTimer = setTimeout(
                        checkInvocation,
                        getInterval(invokeResult)
                    );
                }
            };
        checkInvocation();
    }

    /**
     * Monitors an invocation on the canvas.
     *
     * @param string uuid The UUID of the invocation.
     */
    async canvasInvocation(uuid) {
        let complete = false,
            invocationComplete = false,
            start = (new Date()).getTime(),
            lastTick = start,
            lastTotalChangeTime = start,
            lastStepChangeTime = start,
            lastTotalTime,
            lastEstimate,
            lastEstimateTime,
            lastPercentComplete,
            lastRemainingTime,
            lastImages,
            lastStep,
            lastTotal,
            lastRate,
            lastDuration,
            visibleInvocation = 0,
            invocationSampleChooserImageNodes = [],
            loadedNode = this.loadingBar.find(E.getCustomTag("invocationLoaded")),
            durationNode = this.loadingBar.find(E.getCustomTag("invocationDuration")),
            iterationsNode = this.loadingBar.find(E.getCustomTag("invocationIterations")),
            remainingNode = this.loadingBar.find(E.getCustomTag("invocationRemaining")),
            updateImages = () => {
                if (isEmpty(lastImages)) {
                    this.invocationSampleChooser.hide();
                } else if (invocationSampleChooserImageNodes.length === 0) {
                    this.images.setCurrentInvocationImage(lastImages[0]);
                    this.invocationSampleChooser.empty().append(
                        E.invocationSample().class("no-sample").content("×").on("click", () => {
                            visibleInvocation = null;
                            this.images.hideCurrentInvocation();
                        })
                    );
                    for (let i in lastImages) {
                        let imageNode = E.img().src(lastImages[i]);
                        invocationSampleChooserImageNodes.push(imageNode);
                        this.invocationSampleChooser.append(
                            E.invocationSample().content(imageNode).on("click", () => {
                                this.images.setCurrentInvocationImage(imageNode.src())
                                visibleInvocation = i;
                            })
                        );
                    }
                    this.invocationSampleChooser.show().render();
                } else {
                    for (let i in lastImages) {
                        invocationSampleChooserImageNodes[i].src(lastImages[i]);
                    }
                    if (visibleInvocation !== null) {
                        this.images.setCurrentInvocationImage(lastImages[visibleInvocation]);
                    }
                }
            },
            updateNodes = () => {
                let elapsedTime = lastTick - start;
                durationNode.content(humanDuration(elapsedTime/1000));
                if (isEmpty(lastPercentComplete)) {
                    loadedNode.css("width", "100%");
                    iterationsNode.hide();
                    remainingNode.show().content("Initializing…");
                } else if (lastPercentComplete < 100) {
                    remainingNode.show().content(humanDuration(lastRemainingTime/1000));
                    loadedNode.css("width", `${lastPercentComplete.toFixed(2)}%`);
                    let iterationSpeed = lastRate,
                        iterationUnit = "it/s";
                    if (!isEmpty(iterationSpeed) && iterationSpeed < Infinity) {
                        if (iterationSpeed < 1) {
                            iterationSpeed = 1 / iterationSpeed;
                            iterationUnit = "s/it";
                        }
                        iterationsNode.show().content(`Iteration ${lastStep}/${lastTotal} (${iterationSpeed.toFixed(2)} ${iterationUnit})`);
                    } else {
                        iterationsNode.show().content(`Iteration ${lastStep}/${lastTotal}`);
                    }
                } else if (!complete && !isEmpty(lastDuration)) {
                    remainingNode.content("Finalizing step…");
                    let iterationSpeed = lastTotal / lastDuration,
                        iterationUnit = "it/s";
                    if (iterationSpeed < 1) {
                        iterationSpeed = 1 / iterationSpeed;
                        iterationUnit = "s/it";
                    }

                    iterationsNode.content(`${lastTotal} Iterations at ${iterationSpeed.toFixed(2)} ${iterationUnit}`);
                    loadedNode.css("width", "100%");
                } else {
                    loadedNode.css("width", "0");
                    remainingNode.empty().hide();
                }
            },
            updateEstimate = () => {
                let tickTime = (new Date()).getTime();
                if (invocationComplete) {
                    if (lastPercentComplete < 100 && lastPercentComplete > 0) {
                        lastPercentComplete += 1;
                    } else {
                        lastPercentComplete = 100;
                        complete = true;
                    }
                } else {
                    if (!isEmpty(lastEstimate) && lastEstimate < Infinity) {
                        let elapsedTime = tickTime - start,
                            elapsedTimeThisStep = tickTime - lastTotalChangeTime,
                            remainingTime = lastEstimate - (tickTime - lastEstimateTime),
                            totalTime = elapsedTimeThisStep + remainingTime,
                            percentComplete = (elapsedTimeThisStep/totalTime) * 100;
                        
                        lastRemainingTime = remainingTime;
                        lastTotalTime = totalTime;
                        lastPercentComplete = percentComplete;
                    }
                }
                lastTick = tickTime;
                updateNodes();
                if (!complete) {
                    requestAnimationFrame(() => updateEstimate());
                }
            },
            onImagesReceived = async (images, isComplete) => {
                invocationComplete = isComplete;
                if (images.length > 0) {
                    lastImages = images;
                    updateImages();
                }
            },
            onError = () => {
                invocationComplete = true;
                complete = true;
                remainingNode.empty().hide();
                iterationsNode.empty().hide();
            },
            onEstimatedDuration = (milliseconds, step, total, rate, duration) => {
                lastEstimate = milliseconds;
                if (lastStep !== step) {
                    lastStepChangeTime = lastEstimateTime;
                }
                lastStep = step;
                lastRate = rate;
                lastDuration = duration;
                lastEstimateTime = (new Date()).getTime();
                if (lastTotal !== total) {
                    lastTotalChangeTime = lastEstimateTime;
                    if (!isEmpty(lastTotal)) {
                        lastStep = 0;
                        lastPercentComplete = null;
                        lastStepChangeTime = lastEstimateTime;

                    }
                }
                lastTotal = total;
            };

        this.loadingBar.addClass("loading");
        this.images.hideCurrentInvocation();
        this.invocationSampleChooser.empty();

        window.requestAnimationFrame(() => updateEstimate());
        this.monitorInvocation(uuid, onImagesReceived, onError, onEstimatedDuration);
        await waitFor(() => complete);
        this.loadingBar.removeClass("loading");
    }

    /**
     * Monitors an invocation in a detached window.
     *
     * @param string name The name of the window to spawn.
     * @param string uuid The UUID of the invocation.
     * @param int expectedWidth The width of the invocation.
     * @param int expectedHeight The height of the invocation.
     */
    async startDetachedInvocationMonitor(name, uuid, expectedWidth, expectedHeight) {
        let receivedFirstImage = false,
            invocationWindows = [],
            inspectorViews = [],
            onImagesReceived = async (images, isComplete, duration) => {
                receivedFirstImage = true;
                for (let i in images) {
                    let imagePath = images[i];
                    if (invocationWindows.length <= i) {
                        let newImageView = new ImageInspectorView(
                            this.application.config,
                            imagePath,
                            uuid
                        );
                        let newWindow = await this.spawnWindow(
                            `${name}, sample ${i+1}`,
                            newImageView,
                            expectedWidth + 30,
                            expectedHeight + 90
                        );
                        newImageView.loading();
                        invocationWindows.push(newWindow);
                        inspectorViews.push(newImageView);
                    } else {
                        inspectorViews[i].setImage(imagePath);
                        invocationWindows[i].setName(`${name}, ${duration} elapsed, sample ${i+1}`);
                    }
                }
                if (isComplete) {
                    for (let i in invocationWindows) {
                        invocationWindows[i].setName(
                            `${name} complete in ${duration}, sample ${i+1}`
                        );
                        inspectorViews[i].doneLoading();
                    }
                }
            },
            onError = () => receivedFirstImage = true;
        this.monitorInvocation(uuid, onImagesReceived, onError);
        await waitFor(() => receivedFirstImage === true);
    }
}

export { InvocationController };
