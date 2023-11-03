from __future__ import annotations

import io
import inspect

from contextlib import contextmanager, ExitStack

from PIL.PngImagePlugin import PngInfo

from dataclasses import (
    dataclass,
    asdict,
    field,
)

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    List,
    Callable,
    Iterator,
    TYPE_CHECKING,
)

from pibble.util.strings import Serializer

from enfugue.util import (
    logger,
    feather_mask,
    fit_image,
    save_frames_or_image,
    redact_images_from_metadata,
    merge_tokens,
)

from enfugue.diffusion.constants import *

if TYPE_CHECKING:
    from PIL.Image import Image
    from enfugue.diffusers.manager import DiffusionPipelineManager
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    from enfugue.util import IMAGE_FIT_LITERAL, IMAGE_ANCHOR_LITERAL

__all__ = ["LayeredInvocation"]

@dataclass
class LayeredInvocation:
    """
    A serializable class holding all vars for an invocation
    """
    # Dimensions, required
    width: int
    height: int
    # Model args
    model: Optional[str]=None
    refiner: Optional[str]=None
    inpainter: Optional[str]=None
    vae: Optional[str]=None
    refiner_vae: Optional[str]=None
    inpainter_vae: Optional[str]=None
    lora: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]]=None
    lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]]=None
    inversion: Optional[Union[str, List[str]]]=None
    scheduler: Optional[SCHEDULER_LITERAL]=None
    ip_adapter_plus: bool=False
    ip_adapter_face: bool=False
    # Custom model args
    model_prompt: Optional[str]=None
    model_prompt_2: Optional[str]=None
    model_negative_prompt: Optional[str]=None
    model_negative_prompt_2: Optional[str]=None
    # Invocation
    prompts: Optional[List[PromptDict]]=None
    prompt: Optional[str]=None
    prompt_2: Optional[str]=None
    negative_prompt: Optional[str]=None
    negative_prompt_2: Optional[str]=None
    clip_skip: Optional[int]=None
    tiling_size: Optional[int]=None
    tiling_stride: Optional[int]=None
    tiling_mask_type: Optional[MASK_TYPE_LITERAL]=None
    tiling_mask_kwargs: Optional[Dict[str, Any]]=None
    # Layers
    layers: List[Dict[str, Any]]=field(default_factory=list) #TODO: stronger type
    # Generation
    samples: int=1
    iterations: int=1
    seed: Optional[int]=None
    tile: Union[bool, Tuple[bool, bool], List[bool]]=False
    # Tweaks
    freeu_factors: Optional[Tuple[float, float, float, float]]=None
    guidance_scale: Optional[float]=None
    num_inference_steps: Optional[int]=None
    noise_offset: Optional[float]=None
    noise_method: NOISE_METHOD_LITERAL="perlin"
    noise_blend_method: LATENT_BLEND_METHOD_LITERAL="inject"
    # Animation
    animation_frames: Optional[int]=None
    animation_rate: int=8
    frame_window_size: Optional[int]=16
    frame_window_stride: Optional[int]=4
    loop: bool=False
    motion_module: Optional[str]=None
    motion_scale: Optional[float]=None
    position_encoding_truncate_length: Optional[int]=None
    position_encoding_scale_length: Optional[int]=None
    # img2img
    strength: Optional[float]=None
    # Inpainting
    mask: Optional[Union[Image, str]]=None
    crop_inpaint: bool=True
    inpaint_feather: int=32
    outpaint: bool=True
    # Refining
    refiner_start: Optional[float]=None
    refiner_strength: Optional[float]=None
    refiner_guidance_scale: float=DEFAULT_REFINER_GUIDANCE_SCALE
    refiner_aesthetic_score: float=DEFAULT_AESTHETIC_SCORE
    refiner_negative_aesthetic_score: float=DEFAULT_NEGATIVE_AESTHETIC_SCORE
    refiner_prompt: Optional[str]=None
    refiner_prompt_2: Optional[str]=None
    refiner_negative_prompt: Optional[str]=None
    refiner_negative_prompt_2: Optional[str]=None
    # Flags
    build_tensorrt: bool=False
    # Post-processing
    upscale: Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]]=None
    interpolate_frames: Optional[Union[int, Tuple[int, ...], List[int]]]=None
    reflect: bool=False

    @staticmethod
    def merge_prompts(*args: Tuple[Optional[str], float]) -> str:
        """
        Merges prompts if they are not null
        """
        if all([not prompt for prompt, weight in args]):
            return None
        return merge_tokens(**dict([
            (prompt, weight)
            for prompt, weight in args
            if prompt
        ]))

    @classmethod
    def get_image_bounding_box(
        cls,
        mask: Image,
        size: int,
        feather: int=32
    ) -> List[Tuple[int, int]]:
        """
        Gets the feathered bounding box for an image
        """
        width, height = mask.size
        x0, y0, x1, y1 = mask.getbbox()

        # Add feather
        x0 = max(0, x0 - feather)
        x1 = min(width, x1 + feather)
        y0 = max(0, y0 - feather)
        y1 = min(height, y1 + feather)
        
        # Create centered frame about the bounding box
        bbox_width = x1 - x0
        bbox_height = y1 - y0

        if bbox_width < size:
            x0 = max(0, x0 - ((size - bbox_width) // 2))
            x1 = min(width, x0 + size)
            x0 = max(0, x1 - size)
        if bbox_height < size:
            y0 = max(0, y0 - ((size - bbox_height) // 2))
            y1 = min(height, y0 + size)
            y0 = max(0, y1 - size)

        return [(x0, y0), (x1, y1)]

    @classmethod
    def get_inpaint_bounding_box(
        cls,
        mask: Union[Image, List[Image]],
        size: int,
        feather: int=32
    ) -> List[Tuple[int, int]]:
        """
        Gets the bounding box of places inpainted
        """
        # Find bounding box
        (x0, y0) = (0, 0)
        (x1, y1) = (0, 0)
        for mask_image in (mask if isinstance(mask, list) else [mask]):
            (frame_x0, frame_y0), (frame_x1, frame_y1) = cls.get_image_bounding_box(
                mask=mask_image,
                size=size,
                feather=feather,
            )
            x0 = max(x0, frame_x0)
            y0 = max(y0, frame_y0)
            x1 = max(x1, frame_x1)
            y1 = max(y1, frame_y1)

        return [(x0, y0), (x1, y1)]

    @classmethod
    def paste_inpaint_image(
        cls,
        background: Image,
        foreground: Image,
        position: Tuple[int, int],
        inpaint_feather: int=32,
    ) -> Image:
        """
        Pastes the inpaint image on the background with an appropriately feathered mask.
        """
        from PIL import Image
        image = background.copy()

        width, height = image.size
        foreground_width, foreground_height = foreground.size
        left, top = position[:2]
        right, bottom = left + foreground_width, top + foreground_height

        feather_left = left > 0
        feather_top = top > 0
        feather_right = right < width
        feather_bottom = bottom < height

        mask = Image.new("L", (foreground_width, foreground_height), 255)
        for i in range(inpaint_feather):
            multiplier = (i + 1) / (inpaint_feather + 1)
            pixels = []
            if feather_left:
                pixels.extend([(i, j) for j in range(foreground_height)])
            if feather_top:
                pixels.extend([(j, i) for j in range(foreground_width)])
            if feather_right:
                pixels.extend([(foreground_width - i - 1, j) for j in range(foreground_height)])
            if feather_bottom:
                pixels.extend([(j, foreground_height - i - 1) for j in range(foreground_width)])
            for x, y in pixels:
                mask.putpixel((x, y), int(mask.getpixel((x, y)) * multiplier))

        image.paste(foreground, position, mask=mask)
        return image

    @property
    def upscale_steps(self) -> Iterator[UpscaleStepDict]:
        """
        Iterates over upscale steps.
        """
        if self.upscale is not None:
            if isinstance(self.upscale, list):
                for step in self.upscale:
                    yield step
            else:
                yield self.upscale

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the keyword arguments that will passed to the pipeline invocation.
        """
        return {
            "width": self.width,
            "height": self.height,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "refiner_start": self.refiner_start,
            "refiner_strength": self.refiner_strength,
            "refiner_guidance_scale": self.refiner_guidance_scale,
            "refiner_aesthetic_score": self.refiner_aesthetic_score,
            "refiner_negative_aesthetic_score": self.refiner_negative_aesthetic_score,
            "refiner_prompt": self.refiner_prompt,
            "refiner_prompt_2": self.refiner_prompt_2,
            "refiner_negative_prompt": self.refiner_negative_prompt,
            "refiner_negative_prompt_2": self.refiner_negative_prompt_2,
            "ip_adapter_plus": self.ip_adapter_plus,
            "ip_adapter_face": self.ip_adapter_face,
        }

    @classmethod
    def prepare_image(
        cls,
        width: int,
        height: int,
        image: Union[Image, List[Image]],
        animation_frames: Optional[int]=None,
        fit: Optional[IMAGE_FIT_LITERAL]=None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL]=None,
        x: Optional[int]=None,
        y: Optional[int]=None,
        w: Optional[int]=None,
        h: Optional[int]=None,
        return_mask: bool=True,
    ) -> Union[Image, List[Image], Tuple[Image, Image], Tuple[List[Image], List[Image]]]:
        """
        Fits an image on the canvas and returns it and it's alpha mask
        """
        from PIL import Image

        if w is not None and h is not None:
            fitted_image = fit_image(image, w, h, fit, anchor)
        else:
            fitted_image = fit_image(image, width, height, fit, anchor)

        if x is not None and y is not None:
            if isinstance(fitted_image, list):
                blank_image_list = [
                    Image.new("RGBA", (width, height), (0,0,0,0))
                    for i in range(len(fitted_image))
                ]
                fitted_image = [
                    blank.paste(fit_image, (x, y))
                    for blank, fit_image in zip(blank_image_list, fitted_image)
                ]
            else:
                blank_image = Image.new("RGBA", (width, height), (0,0,0,0))
                blank_image.paste(fitted_image, (x, y))
                fitted_image = blank_image

        if isinstance(fitted_image, list):
            if not animation_frames:
                fitted_image = fitted_image[0]
            else:
                fitted_image = fitted_image[:animation_frames]

        if not return_mask:
            return fitted_image

        if isinstance(fitted_image, list):
            image_mask = [
                Image.new("1", (width, height), (1))
                for i in range(len(fitted_image))
            ]
        else:
            image_mask = Image.new("1", (width, height), (1))

        black = Image.new("1", (width, height), (0))

        if isinstance(fitted_image, list):
            for i, img in enumerate(fitted_image):
                fitted_alpha = img.split()[-1]
                fitted_alpha_inverse_clamp = Image.eval(fitted_alpha, lambda a: 0 if a > 128 else 255)
                image_mask[i].paste(black, mask=fitted_alpha_inverse_clamp)
        else:
            fitted_alpha = fitted_image.split()[-1]
            fitted_alpha_inverse_clamp = Image.eval(fitted_alpha, lambda a: 0 if a > 128 else 255)
            image_mask.paste(black, mask=fitted_alpha_inverse_clamp)

        return fitted_image, image_mask

    @classmethod
    def assemble(
        cls,
        size: int=512,
        **kwargs: Any
    ) -> DiffusionPipelineInvocation:
        """
        Assembles an invocation from layers, standardizing arguments
        """
        invocation_kwargs = dict([
            (k, v) for k, v in kwargs.items()
            if k in inspect.signature(cls).parameters
        ])
        ignored_kwargs = set(list(kwargs.keys())) - set(list(invocation_kwargs.keys()))

        # Add directly passed images to layers
        layers = invocation_kwargs.pop("layers", [])

        if "image" in ignored_kwargs:
            layers.append({"image": kwargs["image"]})
            ignored_kwargs -= {"image"}

        if "ip_adapter_images" in ignored_kwargs:
            for image in kwargs["ip_adapter_images"]:
                if isinstance(image, dict):
                    layers.append(image)
                else:
                    layers.append({"image": image, "ip_adapter_scale": 1.0})
            ignored_kwargs -= {"ip_adapter_images"}

        if "control_images" in ignored_kwargs:
            layers.extend(kwargs["control_images"])
            ignored_kwargs -= {"control_images"}

        # Reassign layers
        invocation_kwargs["layers"] = layers

        # Gather size of images for defaults
        image_width, image_height = 0, 0
        for layer in layers:
            if (
                layer.get("image", None) is not None and
                (
                    layer.get("denoise", False) or
                    (
                        not layer.get("ip_adapter_scale", None) and
                        not layer.get("control_units", [])
                    )
                )
            ):
                layer_x = layer.get("x", 0)
                layer_y = layer.get("y", 0)
                image_w, image_h = layer["image"].size
                layer_w = layer.get("w", image_w)
                layer_h = layer.get("h", image_h)
                image_width = max(image_width, layer_x + layer_w)
                image_height = max(image_height, layer_y + layer_h)

        # Check sizes
        if not invocation_kwargs.get("width", None):
            invocation_kwargs["width"] = image_width if image_width else size
        if not invocation_kwargs.get("height", None):
            invocation_kwargs["height"] = image_height if image_height else size

        if ignored_kwargs:
            logger.warning(f"Ignored keyword arguments: {ignored_kwargs}")

        return cls(**invocation_kwargs)

    @classmethod
    def minimize_dict(
        cls,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pops unnecessary variables from an invocation dict
        """
        all_keys = list(kwargs.keys())
        minimal_keys = []
        for key in all_keys:
            value = kwargs[key]
            if value is None:
                continue
            if "layers" == key and not value:
                continue
            if "tile" == key and value == False:
                continue
            if "refiner" in key and not kwargs.get("refiner", None):
                continue
            if "inpaint" in key and not kwargs.get("mask", None):
                continue
            if "ip_adapter" in key and not kwargs.get("ip_adapter_images", None):
                continue
            if (
                (
                    "motion" in key or
                    "temporal" in key or
                    "animation" in key or
                    "frame" in key or
                    "loop" in key or
                    "reflect" in key
                ) and
                not kwargs.get("animation_frames", None)
            ):
                continue
            if "noise" in key and not kwargs.get("noise_offset", None):
                continue
            minimal_keys.append(key)

        return dict([
            (key, kwargs[key])
            for key in minimal_keys
        ])

    @classmethod
    def format_serialization_dict(
        cls,
        save_directory: Optional[str]=None,
        save_name: Optional[str]=None,
        image: Optional[Union[str, Image, List[Image]]]=None,
        mask: Optional[Union[str, Image, List[Image]]]=None,
        control_images: Optional[Dict[str, List[Dict]]]=None,
        ip_adapter_images: Optional[List[Dict]]=None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Formats kwargs to remove images and instead reutnr temporary paths if possible
        """
        if save_directory is None:
            kwargs["image"] = image
            kwargs["mask"] = mask
        else:
            if image is not None:
                kwargs["image"] = save_frames_or_image(
                    image=image,
                    directory=save_directory,
                    name=save_name
                )
            if mask is not None:
                kwargs["mask"] = save_frames_or_image(
                    image=mask,
                    directory=save_directory,
                    name=f"{save_name}_mask" if save_name is not None else None
                )
            if control_images is not None:
                if isinstance(control_images, dict):
                    for controlnet in control_images:
                        for i, control_dict in enumerate(control_images[controlnet]):
                            control_dict["image"] = save_frames_or_image(
                                image=control_dict["image"],
                                directory=save_directory,
                                name=f"{save_name}_{controlnet}_{i}" if save_name is not None else None
                            )
                else:
                    for i, control_dict in enumerate(control_images):
                        control_dict["image"] = save_frames_or_image(
                            image=control_dict["image"],
                            directory=save_directory,
                            name=f"{save_name}_{controlnet}_{i}" if save_name is not None else None
                        )
            if ip_adapter_images is not None:
                for i, ip_dict in enumerate(ip_adapter_images):
                    ip_dict["image"] = save_frames_or_image(
                        image=ip_dict["image"],
                        directory=save_directory,
                        name=f"{save_name}_ip_{i}" if save_name is not None else None
                    )

        kwargs["control_images"] = control_images
        kwargs["ip_adapter_images"] = ip_adapter_images
        return cls.minimize_dict(kwargs)

    def serialize(
        self,
        save_directory: Optional[str]=None,
        save_name: Optional[str]=None,
    ) -> Dict[str, Any]:
        """
        Assembles self into a serializable dict
        """
        return self.format_serialization_dict(
            save_directory=save_directory,
            save_name=save_name,
            **asdict(self)
        )

    @contextmanager
    def preprocessors(
        self,
        pipeline: DiffusionPipelineManager
    ) -> Iterator[Dict[str, Callable[[Image], Image]]]:
        """
        Gets all preprocessors needed for this invocation
        """
        needs_background_remover = False
        needs_control_processors = []
        to_check: List[Dict[str, Any]] = []

        if self.layers is not None:
            for layer in self.layers:
                if layer.get("image", None) is not None:
                    to_check.append(layer)
        for image_dict in to_check:
            if image_dict.get("remove_background", False):
                needs_background_remover = True
            if image_dict.get("process", False) and image_dict.get("controlnet", None) is not None:
                needs_control_procesors.append(image_dict["controlnet"])

        with ExitStack() as stack:
            processors: Dict[str, Callable[[Image], Image]] = {}
            if needs_background_remover:
                processors["background_remover"] = stack.enter_context(
                    pipeline.background_remover.remover()
                )
            if needs_control_processors:
                processor_names = list(set(needs_control_processors))
                processor_callables = list(pipeline.control_image_processors.processors(*processor_names))
                processors = {**processors, **dict(zip(processor_names, processor_callables))}
            yield processors

    def preprocess(
        self,
        pipeline: DiffusionPipelineManager,
        intermediate_dir: Optional[str]=None,
        raise_when_unused: bool=True,
        task_callback: Optional[Callable[[str], None]]=None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Processes/transforms arguments
        """
        from PIL import Image, ImageOps
        from enfugue.diffusion.util.prompt_util import Prompt

        # Gather images for preprocessing
        control_images = {}
        ip_adapter_images = []
        invocation_mask = None
        invocation_image = None

        if self.layers:
            if task_callback is not None:
                task_callback("Pre-processing layers")

            # Blank images used for merging
            black = Image.new("1", (self.width, self.height), (0))
            white = Image.new("1", (self.width, self.height), (1))

            mask = self.mask

            # Standardize mask
            if mask is not None:
                if isinstance(mask, dict):
                    invert = mask.get("invert", False)
                    mask = mask.get("image", None)
                    if not mask:
                        raise ValueError("Expected mask dictionary to have 'image' key")
                    if invert:
                        from PIL import ImageOps
                        if isinstance(mask, list):
                            mask = [
                                ImageOps.invert(img) for img in mask
                            ]
                        else:
                            mask = ImageOps.invert(mask)

                mask = self.prepare_image(
                    width=self.width,
                    height=self.height,
                    image=mask,
                    animation_frames=self.animation_frames,
                    return_mask=False
                )

            if self.animation_frames:
                invocation_mask = [
                    white.copy()
                    for i in range(self.animation_frames)
                ]
                invocation_image = [
                    black.convert("RGB")
                    for i in range(self.animation_frames)
                ]
            else:
                invocation_image = black.convert("RGB")
                invocation_mask = white.copy()

            # Preprocess images
            with self.preprocessors(pipeline) as processors:
                # Iterate through layers
                for i, layer in enumerate(self.layers):
                    # Basic information for layer
                    w = layer.get("w", None)
                    h = layer.get("h", None)
                    x = layer.get("x", None)
                    y = layer.get("y", None)

                    layer_image = layer.get("image", None)
                    fit = layer.get("fit", None)
                    anchor = layer.get("anchor", None)
                    remove_background = layer.get("remove_background", None)

                    # Capabilities of layer
                    denoise = layer.get("denoise", False)
                    prompt_scale = layer.get("ip_adapter_scale", False)
                    control_units = layer.get("control_units", [])

                    if not layer_image:
                        logger.warning(f"No image, skipping laying {i}")
                        continue

                    if remove_background:
                        if isinstance(layer_image, list):
                            layer_image = [
                                processors["background_remover"](img)
                                for img in layer_image
                            ]
                        else:
                            layer_image = processors["background_remover"](layer_image)

                    fit_layer_image, fit_layer_mask = self.prepare_image(
                        width=self.width,
                        height=self.height,
                        image=layer_image,
                        fit=fit,
                        anchor=anchor,
                        animation_frames=self.animation_frames,
                        w=w,
                        h=h,
                        x=x,
                        y=y
                    )

                    if isinstance(fit_layer_mask, list):
                        inverse_fit_layer_mask = [
                            ImageOps.invert(img)
                            for img in fit_layer_mask
                        ]
                    else:
                        inverse_fit_layer_mask = ImageOps.invert(fit_layer_mask)

                    if denoise:
                        # img2img (maybe still with prompt and control)
                        # Add the layer mask to the overall mask
                        if isinstance(invocation_mask, list):
                            for i, img in enumerate(fit_layer_mask):
                                invocation_mask[i].paste(black, mask=img)
                            for i, img in enumerate(fit_layer_image):
                                invocation_image[i].paste(img, mask=fit_layer_mask[i])
                        else:
                            invocation_image.paste(fit_layer_image, mask=fit_layer_mask)
                    elif not prompt_scale and not control_units:
                        # passthrough only
                        if isinstance(layer_image, list):
                            for i, img in enumerate(fit_layer_image):
                                invocation_image[i].paste(img, mask=fit_layer_mask[i])
                        else:
                            invocation_mask.paste(black, mask=fit_layer_mask)
                            invocation_image.paste(fit_layer_image, mask=fit_layer_mask)

                    if prompt_scale:
                        # ip adapter
                        ip_adapter_images.append({
                            "image": layer_image,
                            "scale": prompt_scale
                        })

                    if control_units:
                        for control_unit in control_units:
                            controlnet = control_unit["controlnet"]

                            if controlnet not in control_images:
                                control_images[controlnet] = []

                            if control_unit.get("process", True):
                                if isinstance(fit_layer_image, list):
                                    control_image = [
                                        processors[controlnet](img)
                                        for img in fit_layer_image
                                    ]
                                else:
                                    control_image = processors[controlnet](fit_layer_image)
                            elif control_unit.get("invert", False):
                                if isinstance(fit_layer_image, list):
                                    control_image = [
                                        ImageOps.invert(img)
                                        for img in fit_layer_image
                                    ]
                                else:
                                    control_image = ImageOps.invert(fit_layer_image)
                            else:
                                control_image = fit_layer_image

                            control_images[controlnet].append({
                                "start": control_unit.get("start", 0.0),
                                "end": control_unit.get("end", 0.0),
                                "scale": control_unit.get("scale", 1.0),
                                "image": control_image
                            })
            if mask:
                # Final mask merge
                mask.paste(white, mask=Image.eval(
                    feather_mask(invocation_mask),
                    lambda a: 0 if a < 128 else 255
                ))
                invocation_mask = mask.convert("L")
            else:
                invocation_mask = feather_mask(invocation_mask).convert("L")

            # Evaluate mask
            (mask_max, mask_min) = invocation_mask.getextrema()
            if mask_max == mask_min == 0:
                # Nothing to do
                if raise_when_unused:
                    raise IOError("Nothing to do - canvas is covered by non-denoised images. Either modify the canvas such that there is blank space to be filled, enable denoising on an image on the canvas, or add inpainting.")
                # Might have no invocation
                invocation_mask = None
            elif mask_max == mask_min == 255:
                # No inpainting
                invocation_mask = None
            elif not self.outpaint and not mask:
                # Disabled outpainting
                invocation_mask = None

        # Evaluate prompts
        prompts = self.prompts
        if prompts is not None:
            # Prompt travel
            prompts = [
                Prompt(
                    positive=self.merge_prompts(
                        (prompt["positive"], 1.0),
                        (self.model_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    positive_2=self.merge_prompts(
                        (prompt.get("positive_2",None), 1.0),
                        (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    negative=self.merge_prompts(
                        (prompt.get("negative",None), 1.0),
                        (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    negative_2=self.merge_prompts(
                        (prompt.get("negative_2",None), 1.0),
                        (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    start=prompt.get("start",None),
                    end=prompt.get("end",None)
                )
                for prompt in prompts
            ]
        elif self.prompt is not None:
            prompts = [
                Prompt(
                    positive=self.merge_prompts(
                        (self.prompt, 1.0),
                        (self.model_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    positive_2=self.merge_prompts(
                        (self.prompt_2, 1.0),
                        (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
                    ),
                    negative=self.merge_prompts(
                        (self.negative_prompt, 1.0),
                        (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
                    ),
                    negative_2=self.merge_prompts(
                        (self.negative_prompt_2, 1.0),
                        (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
                    )
                )
            ]

        # Refiner prompts
        refiner_prompts = {
            "refiner_prompt": self.merge_prompts(
                (self.refiner_prompt, 1.0),
                (self.model_prompt, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_prompt_2": self.merge_prompts(
                (self.refiner_prompt_2, 1.0),
                (self.model_prompt_2, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_negative_prompt": self.merge_prompts(
                (self.refiner_negative_prompt, 1.0),
                (self.model_negative_prompt, MODEL_PROMPT_WEIGHT)
            ),
            "refiner_negative_prompt_2": self.merge_prompts(
                (self.refiner_negative_prompt_2, 1.0),
                (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
            )
        }

        # Completed pre-processing
        results = {**self.minimize_dict({**refiner_prompts, **self.kwargs})}
        if invocation_image:
            results["image"] = invocation_image
            if invocation_mask:
                results["mask"] = invocation_mask
        if control_images:
            results["control_images"] = control_images
        if ip_adapter_images:
            results["ip_adapter_images"] = ip_adapter_images
        if prompts:
            results["prompts"] = prompts
        return results

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[PIL.Image.Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
    ) -> StableDiffusionPipelineOutput:
        """
        This is the main interface for execution.

        The first step will be the one that executes with the selected number of samples,
        and then each subsequent step will be performed on the number of outputs from the
        first step.
        """
        # We import here so this file can be imported by processes without initializing torch
        from diffusers.utils.pil_utils import PIL_INTERPOLATION
        from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

        if task_callback is None:
            task_callback = lambda arg: None
        
        # Set up the pipeline
        pipeline._task_callback = task_callback
        self.prepare_pipeline(pipeline)

        cropped_inpaint_position = None
        background = None
        invocation_kwargs = self.preprocess(
            pipeline,
            raise_when_unused = not self.upscale,
            task_callback=task_callback,
            
        )

        # Determine if we're doing cropped inpainting
        if "mask" in invocation_kwargs and self.crop_inpaint:
            (x0, y0), (x1, y1) = self.get_inpaint_bounding_box(
                invocation_kwargs["mask"],
                size=self.tiling_size if self.tiling_size else 1024 if pipeline.inpainter_is_sdxl else 512,
                feather=self.inpaint_feather
            )
            mask_width, mask_height = invocation_kwargs["mask"].size
            bbox_width = x1 - x0
            bbox_height = y1 - y0
            pixel_ratio = (bbox_height * bbox_width) / (mask_width * mask_height)
            pixel_savings = (1.0 - pixel_ratio) * 100

            if pixel_ratio < 0.75:
                logger.debug(f"Calculated pixel area savings of {pixel_savings:.1f}% by cropping to ({x0}, {y0}), ({x1}, {y1}) ({bbox_width}px by {bbox_height}px)")
                cropped_inpaint_position = (x0, y0, x1, y1)
            else:
                logger.debug(
                    f"Calculated pixel area savings of {pixel_savings:.1f}% are insufficient, will not crop"
                )

        if cropped_inpaint_position is not None:
            # Get copies prior to crop
            if isinstance(invocation_kwargs["image"], list):
                background = [
                    img.copy()
                    for img in invocation_kwargs["image"]
                ]
            else:
                background = invocation_kwargs["image"].copy()
                
            # First wrap callbacks if needed
            if image_callback is not None:
                # Hijack image callback to paste onto background
                original_image_callback = image_callback

                def pasted_image_callback(images: List[PIL.Image.Image]) -> None:
                    """
                    Paste the images then callback.
                    """
                    if isinstance(background, list):
                        images = [
                            self.paste_inpaint_image((background[i] if i < len(background) else background[-1]), image, cropped_inpaint_position) # type: ignore
                            for i, image in enumerate(images)
                        ]
                    else:
                        images = [
                            self.paste_inpaint_image(background, image, cropped_inpaint_position) # type: ignore
                            for image in images
                        ]

                    original_image_callback(images)

                image_callback = pasted_image_callback
            # Now crop images
            if isinstance(invocation_kwargs["image"], list):
                invocation_kwargs["image"] = [
                    img.crop(cropped_inpaint_position)
                    for img in invocation_kwargs["image"]
                ]
                invocation_kwargs["mask"] = [
                    img.crop(cropped_inpaint_position)
                    for img in invocation_kwargs["mask"]
                ]
            else:
                invocation_kwargs["image"] = invocation_kwargs["image"].crop(cropped_inpaint_position)
                invocation_kwargs["mask"] = invocation_kwargs["mask"].crop(cropped_inpaint_position)

            # Also crop control images
            if "control_images" in invocation_kwargs:
                for controlnet in invocation_kwargs["control_images"]:
                    for image_dict in invocation_kwargs[controlnet]:
                        image_dict["image"] = image_dict["image"].crop(cropped_inpaint_position)

            # Assign height and width
            x0, y0, x1, y1 = cropped_inpaint_position
            invocation_kwargs["width"] = x1 - x0
            invocation_kwargs["height"] = y1 - y0

        # Execute primary inference
        images, nsfw = self.execute_inference(
            pipeline,
            task_callback,
            progress_callback,
            image_callback,
            image_callback_steps,
            invocation_kwargs
        )

        if background is not None and cropped_inpaint_position is not None:
            # Paste the image back onto the background
            for i, image in enumerate(images):
                images[i] = self.paste_inpaint_image(
                    background[i] if isinstance(background, list) else background,
                    image,
                    cropped_inpaint_position
                )

        # Execte upscale, if requested
        images, nsfw = self.execute_upscale(
            pipeline,
            images,
            nsfw,
            task_callback,
            progress_callback,
            image_callback,
            image_callback_steps,
            invocation_kwargs
        )

        pipeline.stop_keepalive() # Make sure this is stopped
        return self.format_output(images, nsfw)

    def prepare_pipeline(self, pipeline: DiffusionPipelineManager) -> None:
        """
        Assigns pipeline-level variables.
        """
        pipeline.start_keepalive() # Make sure this is going

        if self.animation_frames is not None and self.animation_frames > 0:
            pipeline.animator = self.model
            pipeline.animator_vae = self.vae
            pipeline.frame_window_size = self.frame_window_size
            pipeline.frame_window_stride = self.frame_window_stride
            pipeline.position_encoding_truncate_length = self.position_encoding_truncate_length
            pipeline.position_encoding_scale_length = self.position_encoding_scale_length
        else:
            pipeline.model = self.model
            pipeline.vae = self.vae

        pipeline.tiling_size = self.tiling_size
        pipeline.tiling_stride = self.tiling_stride
        pipeline.tiling_mask_type = self.tiling_mask_type

        pipeline.refiner = self.refiner
        pipeline.refiner_vae = self.refiner_vae

        pipeline.inpainter = self.inpainter
        pipeline.inpainter_vae = self.inpainter_vae

        pipeline.lora = self.lora
        pipeline.lycoris = self.lycoris
        pipeline.inversion = self.inversion
        pipeline.scheduler = self.scheduler

        if self.build_tensorrt:
            pipeline.build_tensorrt = True

    def execute_inference(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[PIL.Image.Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Executes primary inference
        """
        from PIL import Image, ImageDraw

        # Define progress and latent callback kwargs, we'll add task callbacks ourself later
        callback_kwargs = {
            "progress_callback": progress_callback,
            "latent_callback_steps": image_callback_steps,
            "latent_callback_type": "pil",
            "task_callback": task_callback
        }

        if self.seed is not None:
            # Set up the RNG
            pipeline.seed = self.seed

        total_images = self.iterations
        if self.animation_frames is not None and self.animation_frames > 0:
            total_images *= self.animation_frames
        else:
            total_images *= self.samples

        width = pipeline.size if self.width is None else self.width
        height = pipeline.size if self.height is None else self.height

        images = [
            Image.new("RGBA", (width, height))
            for i in range(total_images)
        ]
        image_draw = [
            ImageDraw.Draw(image)
            for image in images
        ]
        nsfw_content_detected = [False] * total_images

        # Determine what controlnets to use
        controlnets = (
            None if not invocation_kwargs.get("control_images", None)
            else list(invocation_kwargs["control_images"].keys())
        )

        for it in range(self.iterations):
            if image_callback is not None:
                def iteration_image_callback(callback_images: List[Image]) -> None:
                    """
                    Wrap the original image callback so we're actually pasting the initial image on the main canvas
                    """
                    for j, callback_image in enumerate(callback_images):
                        image_index = (it * self.samples) + j
                        images[image_index] = callback_image
                    image_callback(images)  # type: ignore
            else:
                iteration_image_callback = None  # type: ignore

            if invocation_kwargs.get("animation_frames", None):
                pipeline.animator_controlnets = controlnets
            elif invocation_kwargs.get("mask", None):
                pipeline.inpainter_controlnets = controlnets
            else:
                pipeline.controlnets = controlnets

            result = pipeline(
                latent_callback=iteration_image_callback,
                **invocation_kwargs,
                **callback_kwargs
            )

            for j, image in enumerate(result["images"]):
                image_index = (it * self.samples) + j
                images[image_index] = image
                nsfw_content_detected[image_index] = nsfw_content_detected[image_index] or (
                    "nsfw_content_detected" in result and result["nsfw_content_detected"][j]
                )

            # Call the callback
            if image_callback is not None:
                image_callback(images)

        return images, nsfw_content_detected

    def execute_upscale(
        self,
        pipeline: DiffusionPipelineManager,
        images: List[Image],
        nsfw: List[bool],
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[PIL.Image.Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
        invocation_kwargs: Dict[str, Any] = {}
    ) -> Tuple[List[Image], List[bool]]:
        """
        Executes upscale steps
        """
        for upscale_step in self.upscale_steps:
            method = upscale_step["method"]
            amount = upscale_step["amount"]
            num_inference_steps = upscale_step.get("num_inference_steps", DEFAULT_UPSCALE_INFERENCE_STEPS)
            guidance_scale = upscale_step.get("guidance_scale", DEFAULT_UPSCALE_GUIDANCE_SCALE)
            prompt = upscale_step.get("prompt", DEFAULT_UPSCALE_PROMPT)
            prompt_2 = upscale_step.get("prompt_2", None)
            negative_prompt = upscale_step.get("negative_prompt", None)
            negative_prompt_2 = upscale_step.get("negative_prompt_2", None)
            strength = upscale_step.get("strength", None)
            controlnets = upscale_step.get("controlnets", None)
            scheduler = upscale_step.get("scheduler", self.scheduler)
            tiling_stride = upscale_step.get("tiling_stride", DEFAULT_UPSCALE_TILING_STRIDE)
            tiling_size = upscale_step.get("tiling_size", DEFAULT_UPSCALE_TILING_SIZE)
            tiling_mask_type = upscale_step.get("tiling_mask_type", None)
            tiling_mask_kwargs = upscale_step.get("tiling_mask_kwargs", None)
            noise_offset = upscale_step.get("noise_offset", None)
            noise_method = upscale_step.get("noise_method", None)
            noise_blend_method = upscale_step.get("noise_blend_method", None)
            refiner = self.refiner is not None and upscale_step.get("refiner", True)
            
            prompt = self.merge_prompts(
                (prompt, 1.0),
                (self.prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict["positive"], GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            prompt_2 = self.merge_prompts(
                (prompt_2, 1.0),
                (self.prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_prompt_2, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("positive_2", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            negative_prompt = self.merge_prompts(
                (negative_prompt, 1.0),
                (self.negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("negative", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            negative_prompt_2 = self.merge_prompts(
                (negative_prompt_2, 1.0),
                (self.negative_prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT),
                (self.model_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                (self.refiner_negative_prompt_2, MODEL_PROMPT_WEIGHT),
                *[
                    (prompt_dict.get("negative_2", None), GLOBAL_PROMPT_UPSCALE_WEIGHT)
                    for prompt_dict in (self.prompts if self.prompts is not None else [])
                ]
            )

            for i, image in enumerate(images):
                if nsfw is not None and nsfw[i]:
                    logger.debug(f"Image {i} had NSFW content, not upscaling.")
                    continue

                logger.debug(f"Upscaling sample {i} by {amount} using {method}")
                task_callback(f"Upscaling sample {i+1}")

                if method in ["esrgan", "esrganime", "gfpgan"]:
                    if refiner:
                        pipeline.unload_pipeline("clearing memory for upscaler")
                        pipeline.offload_refiner()
                    else:
                        pipeline.offload_pipeline()
                        pipeline.unload_refiner("clearing memory for upscaler")
                    image = pipeline.upscaler(
                        method=method,
                        image=image,
                        tile=512, # Override
                        outscale=amount
                    )
                elif method in PIL_INTERPOLATION:
                    width, height = image.size
                    image = image.resize(
                        (int(width * amount), int(height * amount)),
                        resample=PIL_INTERPOLATION[method]
                    )
                else:
                    logger.error(f"Unknown upscaler {method}")
                    return self.format_output(images, nsfw)

                images[i] = image
                if image_callback is not None:
                    image_callback(images)

            if strength is not None and strength > 0:
                task_callback("Preparing upscale pipeline")

                if refiner:
                    # Refiners have safety disabled from the jump
                    logger.debug("Using refiner for upscaling.")
                    re_enable_safety = False
                    tiling_size = max(tiling_size, pipeline.refiner_size)
                    tiling_stride = min(tiling_stride, pipeline.refiner_size // 2)
                else:
                    # Disable pipeline safety here, it gives many false positives when upscaling.
                    # We'll re-enable it after.
                    logger.debug("Using base pipeline for upscaling.")
                    re_enable_safety = pipeline.safe
                    tiling_size = max(tiling_size, pipeline.size)
                    tiling_stride = min(tiling_stride, pipeline.size // 2)
                    pipeline.safe = False

                if scheduler is not None:
                    pipeline.scheduler = scheduler

                for i, image in enumerate(images):
                    if nsfw is not None and nsfw[i]:
                        logger.debug(f"Image {i} had NSFW content, not upscaling.")
                        continue

                    width, height = image.size
                    kwargs = {
                        "width": width,
                        "height": height,
                        "image": image,
                        "num_results_per_prompt": 1,
                        "prompt": prompt,
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "negative_prompt_2": negative_prompt_2,
                        "strength": strength,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "tiling_size": tiling_size,
                        "tiling_stride": tiling_stride,
                        "tiling_mask_type": tiling_mask_type,
                        "tiling_mask_kwargs": tiling_mask_kwargs,
                        "progress_callback": progress_callback,
                        "latent_callback": image_callback,
                        "latent_callback_type": "pil",
                        "latent_callback_steps": image_callback_steps,
                        "noise_offset": noise_offset,
                        "noise_method": noise_method,
                        "noise_blend_method": noise_blend_method,
                    }

                    if controlnets is not None:
                        if not isinstance(controlnets, list):
                            controlnets = [controlnets] # type: ignore[unreachable]

                        controlnet_names = []
                        controlnet_weights = []

                        for controlnet in controlnets:
                            if isinstance(controlnet, tuple):
                                controlnet, weight = controlnet
                            else:
                                weight = 1.0
                            if controlnet not in controlnet_names:
                                controlnet_names.append(controlnet)
                                controlnet_weights.append(weight)

                        logger.debug(f"Enabling controlnet(s) {controlnet_names} for upscaling")

                        if refiner:
                            pipeline.refiner_controlnets = controlnet_names
                            upscale_pipline = pipeline.refiner_pipeline
                            is_sdxl = pipeline.refiner_is_sdxl
                        else:
                            pipeline.controlnets = controlnet_names
                            upscale_pipeline = pipeline.pipeline
                            is_sdxl = pipeline.is_sdxl

                        controlnet_unique_names = set(controlnet_names)

                        with pipeline.control_image_processor.processors(controlnet_unique_names) as controlnet_processors:
                            controlnet_processor_dict = dict(zip(controlnet_unique_names, controlnet_processors))

                            kwargs["control_images"] = dict([
                                (
                                    controlnet_name,
                                    [(
                                        controlnet_processor_dict[controlnet_name](image),
                                        controlnet_weight
                                    )]
                                )
                                for controlnet_name, controlnet_weight in zip(controlnet_names, controlnet_weights)
                            ])
                    elif refiner:
                        pipeline.refiner_controlnets = None
                        upscale_pipeline = pipeline.refiner_pipeline
                    else:
                        pipeline.controlnets = None
                        upscale_pipeline = pipeline.pipeline

                    logger.debug(f"Upscaling sample {i} with arguments {kwargs}")
                    pipeline.stop_keepalive() # Stop here to kill during upscale diffusion
                    task_callback(f"Re-diffusing Upscaled Sample {i+1}")
                    image = upscale_pipeline(
                        generator=pipeline.generator,
                        device=pipeline.device,
                        offload_models=pipeline.pipeline_sequential_onload,
                        **kwargs
                    ).images[0]
                    pipeline.start_keepalive() # Return keepalive between iterations
                    images[i] = image
                    if image_callback is not None:
                        image_callback(images)
                if re_enable_safety:
                    pipeline.safe = True
                if refiner:
                    logger.debug("Offloading refiner for next inference.")
                    pipeline.refiner_controlnets = None
                    pipeline.offload_refiner()
                else:
                    pipeline.controlnets = None # Make sure we reset controlnets

        return images, nsfw

    def format_output(self, images: List[Image], nsfw: List[bool]) -> StableDiffusionPipelineOutput:
        """
        Adds Enfugue metadata to an image result
        """
        from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
        from PIL import Image
        metadata_dict = self.serialize()
        logger.critical(metadata_dict)
        redact_images_from_metadata(metadata_dict)
        formatted_images = []
        for i, image in enumerate(images):
            byte_io = io.BytesIO()
            metadata = PngInfo()
            metadata.add_text("EnfugueGenerationData", Serializer.serialize(metadata_dict))
            image.save(byte_io, format="PNG", pnginfo=metadata)
            formatted_images.append(Image.open(byte_io))

        return StableDiffusionPipelineOutput(
            images=formatted_images,
            nsfw_content_detected=nsfw
        )