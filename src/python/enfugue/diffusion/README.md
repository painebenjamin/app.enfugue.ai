# The Enfugue Diffusion Pipeline

The class `enfugue.diffusion.pipeline.EnfugueStableDiffusionPipeline` is an extension of `diffusers.pipelines.stable_diffusion.StableDiffusionPipeline`, and accepts most of the same arguments. For help with the `StableDiffusionPipeline` class, it's best to see huggingface's documentation first [here](https://github.com/huggingface/diffusers/tree/main/examples) before understanding how Enfugue's pipeline is special.

## TensorRT

TensorRT support is achieved in an extension of the Enfugue Diffusion pipeline. See [here](https://github.com/painebenjamin/app.enfugue.ai/tree/main/src/python/enfugue/diffusion/rt) for it's documentation.

## Construction Arguments

The initialization signature is as follows:

```python
def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    controlnet: Optional[ControlNetModel],
    scheduler: KarrasDiffusionSchedulers,
    safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPImageProcessor,
    requires_safety_checker: bool = True,
    engine_size: int = 512,
    chunking_size: int = 32,
    chunking_blur: int = 64,
) -> None:
```

Of note is the following:
1. Enfugue allows for an optional ControlNetModel on initialization.
2. Unless chunking is disabled by specifying `chunking_size = 0`, all Enfugue diffusions are 'chunked' (sliced) using a window of the size `engine_size`. This is done to ensure as much compatibility with TensorRT as possible.

## Invocation Arguments

The invocation signature is as follows:

```python
def __call__(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    image: Optional[Union[PIL.Image.Image, str]] = None,
    mask: Optional[Union[PIL.Image.Image, str]] = None,
    control_image: Optional[Union[PIL.Image.Image, str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    chunking_size: Optional[int] = None,
    chunking_blur: Optional[int] = None,
    strength: float = 0.8,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    conditioning_scale: float = 1.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Literal["latent", "pt", "np", "pil"] = "pil",
    return_dict: bool = True,
    scale_image: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    latent_callback: Optional[
        Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]
    ] = None,
    latent_callback_steps: Optional[int] = 1,
    latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[
    StableDiffusionPipelineOutput,
    Tuple[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]], Optional[List[bool]]],
]:
```

Many arguments are present and function precisely the same as the general `StableDiffusionPipeline` arguments. 

Changes of note are as follows:

1. This pipeline does not assume any particular 'mode' of operation, e.g. `txt2img`, `img2img`, etc. It infers the action to be taken based on the passed arguments to this function. Therefore, this method takes a functional superset of arguments of those kinds of pipelines. For instance, passing none of `image`, `mask` and `control_image` assumes basic `txt2img`, just an `image` assumes `img2img`, and `image` and a `mask` assumes `inpainting`, and when additionally using a `control_image` it will use that as input for the `ControlNetModel` passed at initialization.
2. All images can also be passed as strings - in which case they are opened via `PIL.Image.open()`
3. The parameter `scale_image` controls whether or not an image will be scaled to the passed `width` and `height` to make for easy resizing.
4. You can override the initialized `chunking_size` and `chunking_blur` here.
5. There are two callbacks instead of just one. `progress_callback` receives sub-steps in addition to steps (again, to facilitate TensorRT multi-diffusion.) It's signature is `int: current_step, int: total_steps, float: progress
