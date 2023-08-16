import { ImageInspectorView } from "../../view/image.mjs";
import { isEmpty, isEquivalent, waitFor, humanDuration } from "../../base/helpers.mjs";
import { ElementBuilder } from "../../base/builder.mjs";
import { Controller } from "../base.mjs";

const E = new ElementBuilder({
    "invocationTask": "enfugue-invocation-task",
    "invocationLoading": "enfugue-invocation-loading",
    "invocationLoaded": "enfugue-invocation-loaded",
    "invocationDuration": "enfugue-invocation-duration",
    "invocationIterations": "enfugue-invocation-iterations",
    "invocationRemaining": "enfugue-invocation-remaining",
    "invocationSampleChooser": "enfugue-invocation-sample-chooser",
    "invocationSample": "enfugue-invocation-sample",
    "engineStop": "enfugue-engine-stop"
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
     * @return The engine size when not using preconfigured models
     */
    get size() {
        return this.kwargs.size || 512;
    }
    
    /**
     * @param int The engine size when not using preconfigured models
     */
    set size(newSize) {
        if (this.size !== newSize) {
            this.publish("engineSizeChange", newSize);
        }
        this.kwargs.size = newSize;
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
     * @return string Either the configured secondary prompt or empty string.
     */
    get prompt2() {
        return this.kwargs.prompt_2 || "";
    }

    /**
     * @param string newPrompt Sets the prompt to invoke with.
     */
    set prompt2(newPrompt) {
        if (this.prompt2 !== newPrompt) {
            this.publish("enginePrompt2Change", newPrompt);
        }
        this.kwargs.prompt_2 = newPrompt;
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
     * @return string Either the configured secondary negative prompt or empty string.
     */
    get negativePrompt2() {
        return this.kwargs.negative_prompt_2 || "";
    }

    /**
     * @param string newPrompt Sets the secondary negative prompt to invoke with.
     */
    set negativePrompt2(newPrompt) {
        if (this.negativePrompt2 !== newPrompt) {
            this.publish("engineNegativePrompt2Change", newPrompt);
        }
        this.kwargs.negative_prompt_2 = newPrompt;
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
     * @return string The type of model in use, default is checkpoint
     */
    get modelType() {
        return this.kwargs.model_type || "checkpoint";
    }
    
    /**
     * @param string The type of model in use, default is checkpoint
     */
    set modelType(newModelType) {
        if (this.modelType !== newModelType) {
            this.publish("engineModelTypeChange", newModelType);
        }
        this.kwargs.model_type = newModelType;
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
     * @return ?str|array<str> The secondary prompt(s) to use during upscale diffusion. Default null.
     */
    get upscaleDiffusionPrompt2() {
        return this.kwargs.upscale_diffusion_prompt_2;
    }

    /**
     * @param str|array<str> The new secondary prompt(s) to use during upscale diffusion.
     */
    set upscaleDiffusionPrompt2(newPrompt) {
        if (!isEquivalent(this.upscaleDiffusionPrompt2, newPrompt)) {
            this.publish("engineUpscaleDiffusionPrompt2Change", newPrompt);
        }
        this.kwargs.upscale_diffusion_prompt_2 = newPrompt;
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
     * @return ?str|array<str> The secondary prompt(s) to use during upscale diffusion. Default null.
     */
    get upscaleDiffusionNegativePrompt2() {
        return this.kwargs.upscale_diffusion_negative_prompt_2;
    }

    /**
     * @param str|array<str> The new secondary prompt(s) to use during upscale diffusion.
     */
    set upscaleDiffusionNegativePrompt2(newNegativePrompt) {
        if (!isEquivalent(this.upscaleDiffusionNegativePrompt2, newNegativePrompt)) {
            this.publish("engineUpscaleDiffusionNegativePrompt2Change", newNegativePrompt);
        }
        this.kwargs.upscale_diffusion_negative_prompt_2 = newNegativePrompt;
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
     * @return array<string> Optional textual inversion when not using preconfigured models
     */
    get inversion() {
        return this.kwargs.inversion || [];
    }

    /**
     * @param array<string> The new value of inversion for when not using preconfigured models
     */
    set inversion(newInversion) {
        if(!isEquivalent(this.inversion, newInversion)) {
            this.publish("engineInversionChange", newInversion);
        }
        this.kwargs.inversion = newInversion;
    }
    
    /**
     * @return array<object> Optional lora when not using preconfigured models
     */
    get lora() {
        return this.kwargs.lora || [];
    }

    /**
     * @param array<object> The new value of lora for when not using preconfigured models
     */
    set lora(newLora) {
        if(!isEquivalent(this.lora, newLora)) {
            this.publish("engineLoraChange", newLora);
        }
        this.kwargs.lora = newLora;
    }

    /**
     * @return array<object> Optional lycoris when not using preconfigured models
     */
    get lycoris() {
        return this.kwargs.lycoris || [];
    }

    /**
     * @param array<object> The new value of lycoris for when not using preconfigured models
     */
    set lycoris(newLycoris) {
        if(!isEquivalent(this.lycoris, newLycoris)) {
            this.publish("engineLycorisChange", newLycoris);
        }
        this.kwargs.lycoris = newLycoris;
    }

    /**
     * @return string Optional refiner when not using preconfigured models
     */
    get refiner() {
        return this.kwargs.refiner || null;
    }

    /**
     * @param array<object> The new value of refiner for when not using preconfigured models
     */
    set refiner(newRefiner) {
        if(this.refiner !== newRefiner) {
            this.publish("engineRefinerChange", newRefiner);
        }
        this.kwargs.refiner = newRefiner;
    }
    
    /**
     * @return int Optional refining engine size when not using preconfigured models
     */
    get refinerSize() {
        return this.kwargs.refiner_size || null;
    }

    /**
     * @param int Optional inpainting engine size when not using preconfigured models
     */
    set refinerSize(newRefinerSize) {
        if(this.refinerSize !== newRefinerSize) {
            this.publish("engineRefinerSizeChange", newRefinerSize);
        }
        this.kwargs.refiner_size = newRefinerSize;
    }

    /**
     * @return float The strength of the refiner when using SDXL
     */
    get refinerStrength() {
        return this.kwargs.refiner_strength || 0.3;
    }

    /**
     * @param float the new denoising strength to use when refining with SDXL
     */
    set refinerStrength(newRefinerStrength) {
        if(this.refinerStrength !== newRefinerStrength) {
            this.publish("engineRefinerStrengthChange", newRefinerStrength);
        }
        this.kwargs.refiner_strength = newRefinerStrength;
    }
    
    /**
     * @return float The guidance scale of the refiner when using SDXL
     */
    get refinerGuidanceScale() {
        return this.kwargs.refiner_guidance_scale || 5.0
    }

    /**
     * @param float The new guidance scale of the refiner when using SDXL
     */
    set refinerGuidanceScale(newRefinerGuidanceScale) {
        if(this.refinerGuidanceScale !== newRefinerGuidanceScale) {
            this.publish("engineRefinerGuidanceScaleChange", newRefinerGuidanceScale);
        }
        this.kwargs.refiner_guidance_scale = newRefinerGuidanceScale;
    }
    
    /**
     * @return float The aesthetic score of the refiner when using SDXL
     */
    get refinerAestheticScore() {
        return this.kwargs.refiner_aesthetic_score || 6.0
    }

    /**
     * @param float The new aesthetic score
     */
    set refinerAestheticScore(newAestheticScore) {
        if(this.refinerAestheticScore !== newAestheticScore) {
            this.publish("engineAestheticScoreChange", newAestheticScore);
        }
        this.kwargs.refiner_aesthetic_score = newAestheticScore;
    }
    
    /**
     * @return float The negative aesthetic score of the refiner
     */
    get refinerNegativeAestheticScore() {
        return this.kwargs.refiner_negative_aesthetic_score || 2.5;
    }

    /**
     * @param float The new negative aesthetic score of the refiner
     */
    set refinerNegativeAestheticScore(newNegativeAestheticScore) {
        if(this.refinerNegativeAestheticScore !== newNegativeAestheticScore) {
            this.publish("engineNegativeAestheticScoreChange", newNegativeAestheticScore);
        }
        this.kwargs.refiner_negative_aesthetic_score = newNegativeAestheticScore;
    }
    
    /**
     * @return string Optional inpainter when not using preconfigured models
     */
    get inpainter() {
        return this.kwargs.inpainter || null;
    }

    /**
     * @param array<object> The new value of inpainter for when not using preconfigured models
     */
    set inpainter(newInpainter) {
        if(this.inpainter !== newInpainter) {
            this.publish("engineInpainterChange", newInpainter);
        }
        this.kwargs.inpainter = newInpainter;
    }
    
    /**
     * @return int Optional inpainting engine size when not using preconfigured models
     */
    get inpainterSize() {
        return this.kwargs.inpainter_size || null;
    }

    /**
     * @param int Optional inpainting engine size when not using preconfigured models
     */
    set inpainterSize(newInpainterSize) {
        if(this.inpainterSize !== newInpainterSize) {
            this.publish("engineInpainterSizeChange", newInpainterSize);
        }
        this.kwargs.inpainter_size = newInpainterSize;
    }

    /**
     * @return str The scheduler, if set
     */
    get scheduler() {
        return this.kwargs.scheduler || null;
    }
    
    /**
     * @param str Set the new scheduler
     */
    set scheduler(newScheduler) {
        if (this.scheduler !== newScheduler) {
            this.publish("engineSchedulerChange");
        }
        this.kwargs.scheduler = newScheduler;
    }

    /**
     * @return str The multi-diffusion scheduler, if set
     */
    get multiScheduler() {
        return this.kwargs.multi_scheduler || null;
    }
    
    /**
     * @param str Set the new scheduler for multi-diffusion
     */
    set multiScheduler(newScheduler) {
        if (this.multiScheduler !== newScheduler) {
            this.publish("engineMultiDiffusionSchedulerChange");
        }
        this.kwargs.multi_scheduler = newScheduler;
    }

    /**
     * @return str The vae, if set
     */
    get vae() {
        return this.kwargs.vae || null;
    }
    
    /**
     * @param str Set the vae
     */
    set vae(newVae) {
        if (this.vae !== newVae) {
            this.publish("engineVaeChange");
        }
        this.kwargs.vae = newVae;
    }
    
    /**
     * On initialization, create DOM elements related to invocations.
     */
    async initialize() {
        this.loadingBar = E.invocationLoading().content(
            E.invocationLoaded().addClass("sliding-gradient"),
            E.invocationDuration(),
            E.invocationIterations(),
            E.invocationTask().hide(),
            E.invocationRemaining().hide()
        );
        this.invocationSampleChooser = E.invocationSampleChooser().hide();
        this.engineStop = E.engineStop().content("Stop Engine").on("click", () => { this.stopEngine() });
        (await this.images.getNode()).append(this.loadingBar).append(this.invocationSampleChooser);
        this.application.container.appendChild(await this.engineStop.render());
        this.subscribe("engineReady", () => {
            this.enableStop();
        });
        this.subscribe("engineBusy", () => {
            this.enableStop();
        });
        this.subscribe("engineIdle", () => {
            this.disableStop();
        });
    }

    /**
     * Hides the sample chooser from outside the controller.
     */
    hideSampleChooser() {
        this.invocationSampleChooser.hide();
    }

    /**
     * Enables the engine stopper.
     */
    enableStop() {
        this.engineStop.addClass("ready");
    }
    
    /**
     * Disbles the engine stopper.
     */
    disableStop() {
        this.engineStop.removeClass("ready");
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
        this.enableStop();
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
            }
        }
    }

    /**
     * Stops the engine.
     */
    async stopEngine() {
        if (!this.engineStop.hasClass("ready")) {
            return;
        }
        if (await this.confirm("Stop engine and terminate any active invocations?")) {
            try {
                await this.application.model.post("/invocation/stop");
                this.disableStop();
                this.notify("info", "Stopped", "Successfully stopped engine.");
            } catch(e) {
                let errorMessage = `${e}`;
                if (!isEmpty(e.detail)) errorMessage = e.detail;
                else if (!isEmpty(e.title)) errorMessage = e.title;
                this.notify("error", "Error", `Received an error when stopping. The engine may still be stopped, wait a moment and check again. ${errorMessage}`);
            }
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
    async monitorInvocation(uuid, onTaskChanged, onImagesReceived, onError, onEstimatedDuration) {
        const initialInterval = this.application.config.model.invocation.interval || 1000;
        const queuedInterval = this.application.config.model.queue.interval || 5000;
        const consecutiveErrorCutoff = this.application.config.model.invocation.errors.consecutive || 2;

        if (onImagesReceived === undefined) onImagesReceived = () => {};
        if (onTaskChanged === undefined) onTaskChanged = () => {};
        if (onError === undefined) onError = () => {};
        if (onEstimatedDuration === undefined) onEstimatedDuration = () => {};

        let start = (new Date()).getTime(),
            lastTask,
            lastStep,
            lastTotal,
            lastRate,
            lastDuration,
            lastStepDeltaTime,
            lastTotalDeltaTime = start,
            getInterval = (invokeResult) => {
                if (invokeResult.status === "queued") {
                    return queuedInterval;
                }
                return initialInterval;
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
                if (invokeResult.task !== lastTask) {
                    onTaskChanged(invokeResult.task);
                    lastTask = invokeResult.task;
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
                    let averageStepsPerMillisecond = lastStep/(lastStepDeltaTime-lastTotalDeltaTime),
                        currentStepsPerMillisecond = isEmpty(lastRate) ? averageStepsPerMillisecond : lastRate / 1000,
                        weightedStepsPerMillisecond = (currentStepsPerMillisecond * 0.75) + (averageStepsPerMillisecond * 0.25),
                        millisecondsRemainingAtDelta = (lastTotal-lastStep)/weightedStepsPerMillisecond,
                        millisecondsRemainingNow = millisecondsRemainingAtDelta-((new Date()).getTime()-lastStepDeltaTime);
                    if (isNaN(millisecondsRemainingNow)) {
                        millisecondsRemainingNow = Infinity;
                    }
                    onEstimatedDuration(millisecondsRemainingNow, lastStep, lastTotal, lastRate, lastDuration);
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
            lastTask,
            visibleInvocation = 0,
            invocationSampleChooserImageNodes = [],
            taskNode = this.loadingBar.find(E.getCustomTag("invocationTask")),
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
            onTaskChanged = (newTask) => {
                lastTask = newTask;
                if (isEmpty(newTask)) {
                    taskNode.empty().hide();
                } else {
                    taskNode.content(newTask).show();
                }
            },
            onError = () => {
                invocationComplete = true;
                complete = true;
                remainingNode.empty().hide();
                iterationsNode.empty().hide();
                taskNode.empty().hide();
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
        this.monitorInvocation(uuid, onTaskChanged, onImagesReceived, onError, onEstimatedDuration);
        await waitFor(() => complete);
        taskNode.empty().hide();
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
