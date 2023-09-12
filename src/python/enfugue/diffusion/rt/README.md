# Enfugue Stable Diffusion TensorRT Support

Enfugue's TensorRT support was based on [this](https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_tensorrt_txt2img.py) example pipeline in the `diffusers` repository.

The definitions have been expanded and formalized, with some additional features and support.

## Constructor Signature

The `enfugue.diffusion.rt.pipeline.EnfugueTensorRTStableDiffusionPipeline` takes the following construction arguments:

```python
def __init__(
    self,
    vae: AutoencoderKL,
    text_encoder: Optional[CLIPTextModel],
    text_encoder_2: Optional[CLIPTextModelWithProjection],
    tokenizer: Optional[CLIPTokenizer],
    tokenizer_2: Optional[CLIPTokenizer],
    unet: UNet2DConditionModel,
    scheduler: KarrasDiffusionSchedulers,
    safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPImageProcessor,
    requires_safety_checker: bool = True,
    force_zeros_for_empty_prompt: bool = True,
    requires_aesthetic_score: bool = False,
    force_full_precision_vae: bool = False,
    controlnets: Optional[Dict[str, ControlNetModel]] = None,
    ip_adapter: Optional[IPAdapter] = None,
    engine_size: int = 512,  # Recommended even for machines that can handle more
    chunking_size: int = 32,
    chunking_mask_type: MASK_TYPE_LITERAL = "bilinear",
    chunking_mask_kwargs: Dict[str, Any] = {},
    max_batch_size: int = 16,
    force_engine_rebuild: bool = False,
    vae_engine_dir: Optional[str] = None,
    clip_engine_dir: Optional[str] = None,
    unet_engine_dir: Optional[str] = None,
    controlled_unet_engine_dir: Optional[str] = None,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_preview_features: bool = False,
    onnx_opset: int = 17,
) -> None:
```

Of note are the following:
1. Engines are enabled and disabled by passing a directory for their appropriate engine location. This includes all five parameters of `vae_engine_dir`, `clip_engine_dir`, `unet_engine_dir`, `controlled_unet_engine_dir`, and `controlnet_engine_dir`.
2. Some features are enabled or disabled using the `build_` flags. **The default settings are shown, and Enfugue does not permit these to be changed. Change these at your own risk.**
3. The `onnx_opset` is also not able to be set withing Enfugue. The supported opset is tied to the versions of support libraries, so change this at your own risk as well.

## The Five (Six) Engines

Right now, only **two** of the **five** supported engines are built in Enfugue. These are:
1. The `unet`
2. The `controlled_unet`

A third engine is also used, but is not formally defined. The `unet` engine can either have 4 latent dimensions when not inpainting, or 9 when inpainting - so Enfugue also maintains a reference to this third engine and will swap to it when inpainting. When constructing the engine, the number of dimensions is inferred based on the `UNet2DConditionModel`'s configuration.

That means the following are unused:
1. `clip`
2. `vae`
3. `controlnet`

`clip` is disabled not because it doesn't work, but because it's speed gains are negligible compared to the hassle of compilation.
`vae` and `controlnet` work, but these have been difficult to get consistent results with **when doing chunked (sliced) decoding/encoding/inference.**  I welcome anyone's insight on how to make these better in this scenario. They seem to work fine when not chunking, however.

## The Engine Models

The engine models are based off [this](https://github.com/painebenjamin/app.enfugue.ai/blob/main/src/python/enfugue/diffusion/rt/model/base.py) base model, which defines various facts about the inputs and outputs of each network. View the other files in those directories for individual descriptors of their shapes, but the general method of defining one for a model is this:

1. Define your input and output names. Each input/output should match to a single tensor (not an array of tensors.)
2. Determine what inputs and outputs have the same dimensions. Assign each unique tuple of dimensions a key (a short string name of any kind, it can start with a number,) and indicate to the model what those axes ares in `get_dynamic_axes`. This is where TensorRT's magic comes from.
3. Go through the laborious process of defining the exact shape of all input/output tensors based on the dimensions of the model. This is why it's important to define a model size in the beginning, and stick to that model size during inference.
4. Define the `get_sample_input` function to return a random tensor in the shape expected by your definitions above. These random tensors will be pumped into the model to generate it's more optimized format.
