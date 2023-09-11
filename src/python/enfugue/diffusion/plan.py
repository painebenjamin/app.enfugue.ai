from __future__ import annotations

import io
import os
import sys
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import math

from random import randint

from PIL.PngImagePlugin import PngInfo

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
from typing_extensions import (
    TypedDict,
    NotRequired
)

from pibble.util.strings import get_uuid, Serializer

from enfugue.util import (
    logger,
    feather_mask,
    fit_image,
    images_are_equal,
    remove_background as execute_remove_background,
    TokenMerger,
    IMAGE_FIT_LITERAL,
    IMAGE_ANCHOR_LITERAL,
)

if TYPE_CHECKING:
    from enfugue.diffusers.manager import DiffusionPipelineManager
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
    from enfugue.diffusion.constants import (
        SCHEDULER_LITERAL,
        CONTROLNET_LITERAL,
        UPSCALE_LITERAL,
        MASK_TYPE_LITERAL,
    )

DEFAULT_SIZE = 512
DEFAULT_IMAGE_CALLBACK_STEPS = 10
DEFAULT_CONDITIONING_SCALE = 1.0
DEFAULT_IMG2IMG_STRENGTH = 0.8
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_UPSCALE_PROMPT = "highly detailed, ultra-detailed, intricate detail, high definition, HD, 4k, 8k UHD"
DEFAULT_UPSCALE_INFERENCE_STEPS = 100
DEFAULT_UPSCALE_GUIDANCE_SCALE = 12
DEFAULT_UPSCALE_CHUNKING_SIZE = 128

DEFAULT_REFINER_START = 0.85
DEFAULT_REFINER_STRENGTH = 0.3
DEFAULT_REFINER_GUIDANCE_SCALE = 5.0
DEFAULT_AESTHETIC_SCORE = 6.0
DEFAULT_NEGATIVE_AESTHETIC_SCORE = 2.5

MODEL_PROMPT_WEIGHT = 0.2
GLOBAL_PROMPT_STEP_WEIGHT = 0.4
GLOBAL_PROMPT_UPSCALE_WEIGHT = 0.4
UPSCALE_PROMPT_STEP_WEIGHT = 0.1
MAX_IMAGE_SCALE = 3.0

__all__ = ["NodeDict", "DiffusionStep", "DiffusionPlan"]

class UpscaleStepDict(TypedDict):
    method: UPSCALE_LITERAL
    amount: Union[int, float]
    strength: NotRequired[float]
    num_inference_steps: NotRequired[int]
    scheduler: NotRequired[SCHEDULER_LITERAL]
    guidance_scale: NotRequired[float]
    controlnets: NotRequired[List[Union[CONTROLNET_LITERAL, Tuple[CONTROLNET_LITERAL, float]]]]
    prompt: NotRequired[str]
    prompt_2: NotRequired[str]
    negative_prompt: NotRequired[str]
    negative_prompt_2: NotRequired[str]
    chunking_size: NotRequired[int]
    chunking_mask_type: NotRequired[MASK_TYPE_LITERAL]
    chunking_mask_kwargs: NotRequired[Dict[str, Any]]

class ControlImageDict(TypedDict):
    controlnet: CONTROLNET_LITERAL
    image: PIL.Image.Image
    fit: NotRequired[IMAGE_FIT_LITERAL]
    anchor: NotRequired[IMAGE_ANCHOR_LITERAL]
    scale: NotRequired[float]
    process: NotRequired[bool]
    invert: NotRequired[bool]
    refiner: NotRequired[bool]

class NodeDict(TypedDict):
    w: int
    h: int
    x: int
    y: int
    control_images: NotRequired[List[ControlImageDict]]
    image: NotRequired[PIL.Image.Image]
    mask: NotRequired[PIL.Image.Image]
    fit: NotRequired[IMAGE_FIT_LITERAL]
    anchor: NotRequired[IMAGE_ANCHOR_LITERAL]
    prompt: NotRequired[str]
    prompt_2: NotRequired[str]
    negative_prompt: NotRequired[str]
    negative_prompt_2: NotRequired[str]
    strength: NotRequired[float]
    ip_adapter_image: NotRequired[PIL.Image.Image]
    ip_adapter_scale: NotRequired[float]
    remove_background: NotRequired[bool]
    invert_mask: NotRequired[bool]
    crop_inpaint: NotRequired[bool]
    inpaint_feather: NotRequired[int]

class DiffusionStep:
    """
    A step represents most of the inputs to describe what the image is and how to control inference
    """

    result: StableDiffusionPipelineOutput

    def __init__(
        self,
        name: str = "Step", # Can be set later
        width: Optional[int] = None,
        height: Optional[int] = None,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        image: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        mask: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        control_images: Optional[List[ControlImageDict]] = None,
        ip_adapter_scale: Optional[float] = None,
        ip_adapter_image: Optional[float] = None,
        strength: Optional[float] = None,
        num_inference_steps: Optional[int] = DEFAULT_INFERENCE_STEPS,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        refiner_start: Optional[float] = None,
        refiner_strength: Optional[float] = None,
        refiner_guidance_scale: Optional[float] = DEFAULT_REFINER_GUIDANCE_SCALE,
        refiner_aesthetic_score: Optional[float] = DEFAULT_AESTHETIC_SCORE,
        refiner_negative_aesthetic_score: Optional[float] = DEFAULT_NEGATIVE_AESTHETIC_SCORE,
        refiner_prompt: Optional[str] = None,
        refiner_prompt_2: Optional[str] = None,
        refiner_negative_prompt: Optional[str] = None,
        refiner_negative_prompt_2: Optional[str] = None,
        crop_inpaint: Optional[bool] = True,
        inpaint_feather: Optional[int] = None,
        remove_background: bool = False,
        scale_to_model_size: bool = False,
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = negative_prompt_2
        self.image = image
        self.mask = mask
        self.ip_adapter_scale = ip_adapter_scale
        self.ip_adapter_image = ip_adapter_image
        self.control_images = control_images
        self.strength = strength
        self.refiner_start = refiner_start
        self.refiner_strength = refiner_strength
        self.refiner_prompt = refiner_prompt
        self.refiner_prompt_2 = refiner_prompt_2
        self.refiner_negative_prompt = refiner_negative_prompt
        self.refiner_negative_prompt_2 = refiner_negative_prompt_2
        self.remove_background = remove_background
        self.scale_to_model_size = scale_to_model_size
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else DEFAULT_INFERENCE_STEPS
        self.guidance_scale = guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE
        self.refiner_guidance_scale = (
            refiner_guidance_scale if refiner_guidance_scale is not None else DEFAULT_REFINER_GUIDANCE_SCALE
        )
        self.refiner_aesthetic_score = (
            refiner_aesthetic_score if refiner_aesthetic_score is not None else DEFAULT_AESTHETIC_SCORE
        )
        self.refiner_negative_aesthetic_score = (
            refiner_negative_aesthetic_score
            if refiner_negative_aesthetic_score is not None
            else DEFAULT_NEGATIVE_AESTHETIC_SCORE
        )
        self.crop_inpaint = crop_inpaint if crop_inpaint is not None else True
        self.inpaint_feather = inpaint_feather if inpaint_feather is not None else 32

    def get_serialization_dict(self, image_directory: Optional[str]=None) -> Dict[str, Any]:
        """
        Gets the dictionary that will be returned to serialize
        """
        serialized: Dict[str, Any] = {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "prompt_2": self.prompt_2,
            "negative_prompt": self.negative_prompt,
            "negative_prompt_2": self.negative_prompt_2,
            "strength": self.strength,
            "ip_adapter_scale": self.ip_adapter_scale,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "remove_background": self.remove_background,
            "refiner_start": self.refiner_start,
            "refiner_strength": self.refiner_strength,
            "refiner_guidance_scale": self.refiner_guidance_scale,
            "refiner_aesthetic_score": self.refiner_aesthetic_score,
            "refiner_negative_aesthetic_score": self.refiner_negative_aesthetic_score,
            "refiner_prompt": self.refiner_prompt,
            "refiner_prompt_2": self.refiner_prompt_2,
            "refiner_negative_prompt": self.refiner_negative_prompt,
            "refiner_negative_prompt_2": self.refiner_negative_prompt_2,
            "scale_to_model_size": self.scale_to_model_size,
            "crop_inpaint": self.crop_inpaint,
            "inpaint_feather": self.inpaint_feather,
        }

        serialize_children: List[DiffusionStep] = []
        for key in ["image", "mask", "ip_adapter_image"]:
            child = getattr(self, key)
            if isinstance(child, DiffusionStep):
                if child in serialize_children:
                    serialized[key] = serialize_children.index(child)
                else:
                    serialize_children.append(child)
                    serialized[key] = len(serialize_children) - 1
            elif child is not None and image_directory is not None:
                path = os.path.join(image_directory, f"{get_uuid()}.png")
                child.save(path)
                serialized[key] = path
            else:
                serialized[key] = child
        if self.control_images:
            control_images = []
            for control_image in self.control_images:
                image_dict = {
                    "controlnet": control_image["controlnet"],
                    "scale": control_image.get("scale", 1.0),
                    "fit": control_image.get("fit", None),
                    "anchor": control_image.get("anchor", None),
                }
                if isinstance(control_image["image"], DiffusionStep):
                    if control_image["image"] in serialize_children:
                        image_dict["image"] = serialize_children.index(control_image["image"])
                    else:
                        serialize_children.append(control_image["image"])
                        image_dict["image"] = len(serialize_children) - 1
                elif control_image["image"] is not None and image_directory is not None:
                    path = os.path.join(image_directory, f"{get_uuid()}.png")
                    control_image["image"].save(path)
                    image_dict["image"] = path # type:ignore[assignment]
                else:
                    image_dict["image"] = control_image["image"]
                control_images.append(image_dict)
            serialized["control_images"] = control_images
        serialized["children"] = [child.get_serialization_dict(image_directory) for child in serialize_children]
        return serialized

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the keyword arguments that will passed to the pipeline invocation.
        """
        return {
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "prompt_2": self.prompt_2,
            "negative_prompt": self.negative_prompt,
            "negative_prompt_2": self.negative_prompt_2,
            "image": self.image,
            "strength": self.strength,
            "ip_adapter_scale": self.ip_adapter_scale,
            "ip_adapter_image": self.ip_adapter_image,
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
        }

    def get_inpaint_bounding_box(self, pipeline_size: int) -> List[Tuple[int, int]]:
        """
        Gets the bounding box of places inpainted
        """
        if isinstance(self.mask, str):
            mask = PIL.Image.open(self.mask)
        elif isinstance(self.mask, PIL.Image.Image):
            mask = self.mask
        else:
            raise ValueError("Cannot get bounding box for empty or dynamic mask.")
        
        width, height = mask.size
        x0, y0, x1, y1 = mask.getbbox()

        # Add feather
        x0 = max(0, x0 - self.inpaint_feather)
        x1 = min(width, x1 + self.inpaint_feather)
        y0 = max(0, y0 - self.inpaint_feather)
        y1 = min(height, y1 + self.inpaint_feather)
        
        # Create centered frame about the bounding box
        bbox_width = x1 - x0
        bbox_height = y1 - y0

        if bbox_width < pipeline_size:
            x0 = max(0, x0 - ((pipeline_size - bbox_width) // 2))
            x1 = min(width, x0 + pipeline_size)
            x0 = max(0, x1 - pipeline_size)
        if bbox_height < pipeline_size:
            y0 = max(0, y0 - ((pipeline_size - bbox_height) // 2))
            y1 = min(height, y0 + pipeline_size)
            y0 = max(0, y1 - pipeline_size)

        return [(x0, y0), (x1, y1)]

    def paste_inpaint_image(
        self, background: PIL.Image.Image, foreground: PIL.Image.Image, position: Tuple[int, int]
    ) -> PIL.Image.Image:
        """
        Pastes the inpaint image on the background with an appropriately feathered mask.
        """
        image = background.copy()

        width, height = image.size
        foreground_width, foreground_height = foreground.size
        left, top = position
        right, bottom = left + foreground_width, top + foreground_height

        feather_left = left > 0
        feather_top = top > 0
        feather_right = right < width
        feather_bottom = bottom < height

        mask = PIL.Image.new("L", (foreground_width, foreground_height), 255)

        for i in range(self.inpaint_feather):
            multiplier = (i + 1) / (self.inpaint_feather + 1)
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

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        use_cached: bool = True,
        **kwargs: Any,
    ) -> StableDiffusionPipelineOutput:
        """
        Executes this pipeline step.
        """
        if hasattr(self, "result") and use_cached:
            return self.result

        samples = kwargs.pop("samples", 1)

        if isinstance(self.image, DiffusionStep):
            image = self.image.execute(pipeline, samples=1, **kwargs)["images"][0]
        elif isinstance(self.image, str):
            image = PIL.Image.open(self.image)
        else:
            image = self.image

        if isinstance(self.ip_adapter_image, DiffusionStep):
            ip_adapter_image = self.ip_adapter_image.execute(pipeline, samples=1, **kwargs)["images"][0] # type: ignore[unreachable]
        elif isinstance(self.ip_adapter_image, str):
            ip_adapter_image = PIL.Image.open(self.ip_adapter_image) # type: ignore[unreachable]
        else:
            ip_adapter_image = self.ip_adapter_image

        if isinstance(self.mask, DiffusionStep):
            mask = self.mask.execute(pipeline, samples=1, **kwargs)["images"][0]
        elif isinstance(self.mask, str):
            mask = PIL.Image.open(self.mask)
        else:
            mask = self.mask

        if self.control_images is not None:
            control_images: Dict[str, List[Tuple[PIL.Image.Image, float]]] = {}
            for control_image_dict in self.control_images:
                control_image = control_image_dict["image"]
                controlnet = control_image_dict["controlnet"]

                if isinstance(control_image, DiffusionStep):
                    control_image = control_image.execute(pipeline, samples=1, **kwargs)["images"][0]
                elif isinstance(control_image, str):
                    control_image = PIL.Image.open(control_image)

                conditioning_scale = control_image_dict.get("scale", 1.0)
                if control_image_dict.get("process", True):
                    control_image = pipeline.control_image_processor(controlnet, control_image)
                elif control_image_dict.get("invert", False):
                    control_image = PIL.ImageOps.invert(control_image)

                if controlnet not in control_images:
                    control_images[controlnet] = [] # type: ignore[assignment]

                control_images[controlnet].append((control_image, conditioning_scale)) # type: ignore[arg-type]
        else:
            control_images = None # type: ignore[assignment]

        if not self.prompt and not mask and not control_images and not ip_adapter_image and not self.ip_adapter_scale:
            if image:
                if self.remove_background:
                    image = execute_remove_background(image)

                samples = kwargs.get("num_images_per_prompt", 1)
                from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

                self.result = StableDiffusionPipelineOutput(
                    images=[image] * samples, nsfw_content_detected=[False] * samples
                )
                return self.result
            raise ValueError("No prompt or image in this step; cannot invoke or pass through.")

        invocation_kwargs = {**kwargs, **self.kwargs}

        image_scale = 1
        pipeline_size = pipeline.inpainter_size if mask is not None else pipeline.size
        image_width, image_height, image_background, image_position = None, None, None, None

        if image is not None:
            image_width, image_height = image.size
            invocation_kwargs["image"] = image

        if mask is not None:
            mask_width, mask_height = mask.size
            if (
                self.crop_inpaint
                and (mask_width > pipeline_size or mask_height > pipeline.size)
                and image is not None
            ):
                (x0, y0), (x1, y1) = self.get_inpaint_bounding_box(pipeline_size)

                bbox_width = x1 - x0
                bbox_height = y1 - y0

                pixel_ratio = (bbox_height * bbox_width) / (mask_width * mask_height)
                pixel_savings = (1.0 - pixel_ratio) * 100
                if pixel_ratio < 0.75:
                    logger.debug(f"Calculated pixel area savings of {pixel_savings:.1f}% by cropping to ({x0}, {y0}), ({x1}, {y1}) ({bbox_width}px by {bbox_height}px)")
                    # Disable refining
                    invocation_kwargs["refiner_strength"] = 0
                    invocation_kwargs["refiner_start"] = 1
                    image_position = (x0, y0)
                    image_background = image.copy()
                    image = image.crop((x0, y0, x1, y1))
                    mask = mask.crop((x0, y0, x1, y1))
                    image_width, image_height = bbox_width, bbox_height
                    invocation_kwargs["image"] = image  # Override what was set above
                else:
                    logger.debug(
                        f"Calculated pixel area savings of {pixel_savings:.1f}% are insufficient, will not crop"
                    )
            invocation_kwargs["mask"] = mask
            if image is not None:
                assert image.size == mask.size
            else:
                image_width, image_height = mask.size

        if isinstance(ip_adapter_image, PIL.Image.Image):
            if image_width is None or image_height is None:
                image_width, image_height = ip_adapter_image.size
            else:
                image_prompt_width, image_prompt_height = ip_adapter_image.size
                assert image_prompt_width == image_width and image_prompt_height == image_height

        if control_images is not None:
            for controlnet_name in control_images:
                for control_image, conditioning_scale in control_images[controlnet_name]:
                    if image_width is None or image_height is None:
                        image_width, image_height = control_image.size
                    else:
                        this_width, this_height = control_image.size
                        assert image_width == this_width and image_height == this_height
            invocation_kwargs["control_images"] = control_images
            if mask is not None:
                pipeline.inpainter_controlnets = list(control_images.keys())
            else:
                pipeline.controlnets = list(control_images.keys())

        if self.width is not None and self.height is not None and image_width is None and image_height is None:
            image_width, image_height = self.width, self.height

        if image_width is None or image_height is None:
            logger.warning("No known invocation size, defaulting to engine size")
            image_width, image_height = pipeline_size, pipeline_size

        if image_width is not None and image_width < pipeline_size:
            image_scale = pipeline_size / image_width
        if image_height is not None and image_height < pipeline_size:
            image_scale = max(image_scale, pipeline_size / image_height)

        if image_scale > MAX_IMAGE_SCALE or not self.scale_to_model_size:
            # Refuse it's too oblong. We'll just calculate at the appropriate size.
            image_scale = 1

        invocation_kwargs["width"] = 8 * math.ceil((image_width * image_scale) / 8)
        invocation_kwargs["height"] = 8 * math.ceil((image_height * image_scale) / 8)

        if image_scale > 1:
            # scale input images up
            for key in ["image", "mask", "image_mask_image"]:
                if invocation_kwargs.get(key, None) is not None:
                    invocation_kwargs[key] = self.scale_image(invocation_kwargs[key], image_scale)
            for controlnet_name in invocation_kwargs.get("control_images", {}):
                for i, (control_image, conditioning_scale) in enumerate(invocation_kwargs["control_images"].get(controlnet_name, [])):
                    invocation_kwargs["control_images"][controlnet_name][i] = (
                        self.scale_image(control_image, image_scale),
                        conditioning_scale
                    )

        latent_callback = invocation_kwargs.get("latent_callback", None)
        if image_background is not None and image_position is not None and latent_callback is not None:
            # Hijack latent callback to paste onto background
            def pasted_latent_callback(images: List[PIL.Image.Image]) -> None:
                images = [
                    self.paste_inpaint_image(image_background, image, image_position) # type: ignore
                    for image in images
                ]
                latent_callback(images)

            invocation_kwargs["latent_callback"] = pasted_latent_callback

        result = pipeline(**invocation_kwargs)

        if image_background is not None and image_position is not None:
            for i, image in enumerate(result["images"]):
                result["images"][i] = self.paste_inpaint_image(image_background, image, image_position)

        if self.remove_background:
            for i, image in enumerate(result["images"]):
                result["images"][i] = execute_remove_background(image)

        if image_scale > 1:
            for i, image in enumerate(result["images"]):
                result["images"][i] = self.scale_image(image, 1 / image_scale)

        self.result = result
        return result

    @staticmethod
    def scale_image(image: PIL.Image.Image, scale: Union[int, float]) -> PIL.Image.Image:
        """
        Scales an image proportionally.
        """
        width, height = image.size
        scaled_width = 8 * round((width * scale) / 8)
        scaled_height = 8 * round((height * scale) / 8)
        return image.resize((scaled_width, scaled_height))

    @staticmethod
    def deserialize_dict(step_dict: Dict[str, Any]) -> DiffusionStep:
        """
        Given a serialized dict, instantiate a diffusion step
        """
        kwargs: Dict[str, Any] = {}
        for key in [
            "name",
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "strength",
            "num_inference_steps",
            "guidance_scale",
            "ip_adapter_scale",
            "refiner_start",
            "refiner_strength",
            "refiner_guidance_scale",
            "refiner_aesthetic_score",
            "refiner_negative_aesthetic_score",
            "refiner_prompt",
            "refiner_prompt_2",
            "refiner_negative_prompt",
            "refiner_negative_prompt_2",
            "width",
            "height",
            "remove_background",
            "scale_to_model_size",
            "crop_inpaint",
            "inpaint_feather"
        ]:
            if key in step_dict:
                kwargs[key] = step_dict[key]

        deserialized_children = [DiffusionStep.deserialize_dict(child) for child in step_dict.get("children", [])]
        for key in ["image", "mask", "ip_adapter_image"]:
            if key not in step_dict:
                continue
            if isinstance(step_dict[key], int):
                kwargs[key] = deserialized_children[step_dict[key]]

            elif isinstance(step_dict[key], str) and os.path.exists(step_dict[key]):
                kwargs[key] = PIL.Image.open(step_dict[key])
            else:
                kwargs[key] = step_dict[key]
        if "control_images" in step_dict:
            control_images: List[Dict[str, Any]] = []
            for control_image_dict in step_dict["control_images"]:
                control_image = control_image_dict["image"]
                if isinstance(control_image, int):
                    control_image = deserialized_children[control_image]
                elif isinstance(control_image, str):
                    control_image = PIL.Image.open(control_image)
                control_images.append({
                    "image": control_image,
                    "controlnet": control_image_dict["controlnet"],
                    "scale": control_image_dict.get("scale", 1.0),
                    "process": control_image_dict.get("process", True),
                    "invert": control_image_dict.get("invert", False)
                })
            kwargs["control_images"] = control_images

        return DiffusionStep(**kwargs)


class DiffusionNode:
    """
    A diffusion node has a step that may be recursive, combined with bounds.
    """

    def __init__(self, bounds: List[Tuple[int, int]], step: DiffusionStep) -> None:
        self.bounds = bounds
        self.step = step

    def resize_image(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Resizes the image to fit the bounds.
        """
        x, y = self.bounds[0]
        w, h = self.bounds[1]
        return image.resize((w - x, h - y))

    def get_serialization_dict(self, image_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets the step's dict and adds bounds.
        """
        step_dict = self.step.get_serialization_dict(image_directory)
        step_dict["bounds"] = self.bounds
        return step_dict

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        **kwargs: Any,
    ) -> StableDiffusionPipelineOutput:
        """
        Passes through the execution to the step.
        """
        return self.step.execute(pipeline, **kwargs)

    @property
    def name(self) -> str:
        """
        Pass-through the step name
        """
        return self.step.name

    @staticmethod
    def deserialize_dict(step_dict: Dict[str, Any]) -> DiffusionNode:
        """
        Given a serialized dict, instantiate a diffusion Node
        """
        bounds = step_dict.pop("bounds", None)
        if bounds is None:
            raise TypeError("Bounds are required")

        return DiffusionNode(
            [(int(bounds[0][0]), int(bounds[0][1])), (int(bounds[1][0]), int(bounds[1][1]))],
            DiffusionStep.deserialize_dict(step_dict),
        )


class DiffusionPlan:
    """
    A diffusion plan represents any number of steps, with each step receiving the output of the previous.

    Additionally, we handle upscaling as part of the plan. If we want to upscale later, the Plan can be initiated
    with an empty steps array and initial image.
    """

    def __init__(
        self,
        prompt: Optional[str] = None,  # Global
        prompt_2: Optional[str] = None, # Global
        negative_prompt: Optional[str] = None,  # Global
        negative_prompt_2: Optional[str] = None, # Global
        size: Optional[int] = None,
        refiner_size: Optional[int] = None,
        inpainter_size: Optional[int] = None,
        model: Optional[str] = None,
        refiner: Optional[str] = None,
        inpainter: Optional[str] = None,
        lora: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        scheduler: Optional[SCHEDULER_LITERAL] = None,
        vae: Optional[str] = None,
        refiner_vae: Optional[str] = None,
        inpainter_vae: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        nodes: List[DiffusionNode] = [],
        image: Optional[Union[str, PIL.Image.Image]] = None,
        chunking_size: Optional[int] = None,
        chunking_mask_type: Optional[MASK_TYPE_LITERAL] = None,
        chunking_mask_kwargs: Optional[Dict[str, Any]] = None,
        samples: Optional[int] = 1,
        iterations: Optional[int] = 1,
        seed: Optional[int] = None,
        build_tensorrt: bool = False,
        outpaint: bool = True,
        upscale_steps: Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]] = None,
    ) -> None:
        self.size = size if size is not None else (1024 if model is not None and "xl" in model.lower() else 512)
        self.inpainter_size = inpainter_size
        self.refiner_size = refiner_size
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = negative_prompt_2
        self.model = model
        self.refiner = refiner
        self.inpainter = inpainter
        self.lora = lora
        self.lycoris = lycoris
        self.inversion = inversion
        self.scheduler = scheduler
        self.vae = vae
        self.refiner_vae = refiner_vae
        self.inpainter_vae = inpainter_vae
        self.width = width if width is not None else self.size
        self.height = height if height is not None else self.size
        self.image = image
        self.chunking_size = chunking_size if chunking_size is not None else self.size // 8  # Pass 0 to disable
        self.chunking_mask_type = chunking_mask_type
        self.chunking_mask_kwargs = chunking_mask_kwargs
        self.samples = samples if samples is not None else 1
        self.iterations = iterations if iterations is not None else 1
        self.seed = seed if seed is not None else randint(1, sys.maxsize)

        self.outpaint = outpaint
        self.build_tensorrt = build_tensorrt
        self.nodes = nodes
        self.upscale_steps = upscale_steps

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the keyword arguments that will be passing to the pipeline call.
        """
        return {
            "width": self.width,
            "height": self.height,
            "chunking_size": self.chunking_size,
            "chunking_mask_type": self.chunking_mask_type,
            "chunking_mask_kwargs": self.chunking_mask_kwargs,
            "num_images_per_prompt": self.samples,
        }

    @property
    def upscale(self) -> Iterator[UpscaleStepDict]:
        """
        Iterates over upscale steps.
        """
        if self.upscale_steps is not None:
            if isinstance(self.upscale_steps, list):
                for step in self.upscale_steps:
                    yield step
            else:
                yield self.upscale_steps

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

        images, nsfw = self.execute_nodes(
            pipeline,
            task_callback,
            progress_callback,
            image_callback,
            image_callback_steps
        )

        for upscale_step in self.upscale:
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
            chunking_size = upscale_step.get("chunking_size", DEFAULT_UPSCALE_CHUNKING_SIZE)
            scheduler = upscale_step.get("scheduler", self.scheduler)
            chunking_mask_type = upscale_step.get("chunking_mask_type", None)
            chunking_mask_kwargs = upscale_step.get("chunking_mask_kwargs", None)
            refiner = self.refiner is not None and upscale_step.get("refiner", True)

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

                if method == "esrgan":
                    image = pipeline.upscaler.esrgan(image, tile=pipeline.size, outscale=amount)
                elif method == "esrganime":
                    image = pipeline.upscaler.esrgan(image, tile=pipeline.size, outscale=amount, anime=True)
                elif method == "gfpgan":
                    image = pipeline.upscaler.gfpgan(image, tile=pipeline.size, outscale=amount)
                elif method in PIL_INTERPOLATION:
                    width, height = image.size
                    image = image.resize(
                        (int(width * amount), int(height * amount)),
                        resample=PIL_INTERPOLATION[method]
                    )
                else:
                    logger.error(f"Unknown upscaler {method}")
                    return images

                images[i] = image
                if image_callback is not None:
                    image_callback(images)

            if strength is not None and strength > 0:
                task_callback("Preparing upscale pipeline")

                if refiner:
                    # Refiners have safety disabled from the jump
                    logger.debug("Using refiner for upscaling.")
                    re_enable_safety = False
                    chunking_size = min(chunking_size, pipeline.refiner_size // 2)
                    pipeline.reload_refiner()
                else:
                    # Disable pipeline safety here, it gives many false positives when upscaling.
                    # We'll re-enable it after.
                    logger.debug("Using base pipeline for upscaling.")
                    re_enable_safety = pipeline.safe
                    chunking_size = min(chunking_size, pipeline.size // 2)
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
                        "num_images_per_prompt": 1,
                        "prompt": prompt,
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "negative_prompt_2": negative_prompt_2,
                        "strength": strength,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "chunking_size": chunking_size,
                        "chunking_mask_type": chunking_mask_type,
                        "chunking_mask_kwargs": chunking_mask_kwargs,
                        "progress_callback": progress_callback,
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
                            pipeline.reload_refiner()
                            upscale_pipline = pipeline.refiner_pipeline
                            is_sdxl = pipeline.refiner_is_sdxl
                        else:
                            pipeline.controlnets = controlnet_names
                            pipeline.reload_pipeline()
                            upscale_pipeline = pipeline.pipeline
                            is_sdxl = pipeline.is_sdxl

                        kwargs["control_images"] = dict([
                            (
                                controlnet_name,
                                [(
                                    pipeline.control_image_processor(controlnet_name, image),
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
                        pipeline.reload_pipeline()  # If we didn't change controlnet, then pipeline is still on CPU
                        upscale_pipeline = pipeline.pipeline

                    logger.debug(f"Upscaling sample {i} with arguments {kwargs}")
                    pipeline.stop_keepalive() # Stop here to kill during upscale diffusion
                    task_callback(f"Re-diffusing Upscaled Sample {i+1}")
                    image = upscale_pipeline(**kwargs).images[0]
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
        pipeline.stop_keepalive() # Make sure this is stopped
        return self.format_output(images, nsfw)

    def get_image_metadata(self, image: PIL.Image.Image) -> Dict[str, Any]:
        """
        Gets metadata from an image
        """
        (width, height) = image.size
        return {
            "width": width,
            "height": height,
            "metadata": getattr(image, "text", {})
        }

    def redact_images_from_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Removes images from a metadata dictionary
        """
        for key in ["image", "ip_adapter_image", "mask"]:
            if isinstance(metadata.get(key, None), PIL.Image.Image):
                metadata[key] = self.get_image_metadata(metadata[key])
        if "control_images" in metadata:
            for i, control_dict in enumerate(metadata["control_images"]):
                control_dict["image"] = self.get_image_metadata(control_dict["image"])
        if "children" in metadata:
            for child in metadata["children"]:
                self.redact_images_from_metadata(child)
        if "nodes" in metadata:
            for child in metadata["nodes"]:
                self.redact_images_from_metadata(child)

    def format_output(self, images: List[PIL.Image.Image], nsfw: List[bool]) -> StableDiffusionPipelineOutput:
        """
        Adds Enfugue metadata to an image result
        """
        from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

        metadata_dict = self.get_serialization_dict()
        self.redact_images_from_metadata(metadata_dict)
        formatted_images = []
        for i, image in enumerate(images):
            byte_io = io.BytesIO()
            metadata = PngInfo()
            metadata.add_text("EnfugueGenerationData", Serializer.serialize(metadata_dict))
            image.save(byte_io, format="PNG", pnginfo=metadata)
            formatted_images.append(PIL.Image.open(byte_io))

        return StableDiffusionPipelineOutput(
            images=formatted_images,
            nsfw_content_detected=nsfw
        )

    def prepare_pipeline(self, pipeline: DiffusionPipelineManager) -> None:
        """
        Assigns pipeline-level variables.
        """
        pipeline.start_keepalive() # Make sure this is going
        pipeline.model = self.model
        pipeline.refiner = self.refiner
        pipeline.inpainter = self.inpainter
        pipeline.lora = self.lora
        pipeline.lycoris = self.lycoris
        pipeline.inversion = self.inversion
        pipeline.size = self.size
        pipeline.scheduler = self.scheduler
        pipeline.vae = self.vae
        pipeline.refiner_vae = self.refiner_vae
        pipeline.refiner_size = self.refiner_size
        pipeline.inpainter_vae = self.inpainter_vae
        pipeline.inpainter_size = self.inpainter_size
        if self.build_tensorrt:
            pipeline.build_tensorrt = True

    def execute_nodes(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Callable[[str], None],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[PIL.Image.Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
    ) -> Tuple[List[PIL.Image.Image], List[bool]]:
        """
        This is called during execute(). It will go through the steps in order and return
        the resulting image(s).
        """
        if not self.nodes:
            if not self.image:
                raise ValueError("No image and no steps; cannot execute plan.")
            return [self.image], [False]

        # Define progress and latent callback kwargs, we'll add task callbacks ourself later
        callback_kwargs = {
            "progress_callback": progress_callback,
            "latent_callback_steps": image_callback_steps,
            "latent_callback_type": "pil",
        }

        # Set up the pipeline
        self.prepare_pipeline(pipeline)

        if self.seed is not None:
            # Set up the RNG
            pipeline.seed = self.seed

        images = [PIL.Image.new("RGBA", (self.width, self.height)) for i in range(self.samples * self.iterations)]
        image_draw = [PIL.ImageDraw.Draw(image) for image in images]
        nsfw_content_detected = [False] * self.samples * self.iterations

        # Keep a final mask of all nodes to outpaint in the end
        outpaint_mask = PIL.Image.new("RGB", (self.width, self.height), (255, 255, 255))
        outpaint_draw = PIL.ImageDraw.Draw(outpaint_mask)

        for i, node in enumerate(self.nodes):
            def node_task_callback(task: str) -> None:
                """
                Wrap the task callback so we indicate what node we're on
                """
                task_callback(f"{node.name}: {task}")

            invocation_kwargs = {**self.kwargs, **callback_kwargs}
            invocation_kwargs["task_callback"] = node_task_callback
            this_intention = "inpainting" if node.step.mask is not None else "inference"
            next_intention: Optional[str] = None

            if i < len(self.nodes) - 2:
                next_node = self.nodes[i+1]
                next_intention = "inpainting" if next_node.step.mask is not None else "inference"
            elif self.upscale_steps is not None and not (isinstance(self.upscale_steps, list) and len(self.upscale_steps) == 0):
                upscale_step = self.upscale_steps
                if isinstance(upscale_step, list):
                    upscale_step = upscale_step[0]

                upscale_strength = upscale_step.get("strength", None)
                use_ai_upscaler = "gan" in upscale_step["method"]
                use_sd_upscaler = upscale_strength is not None and upscale_strength > 0

                if use_ai_upscaler:
                    next_intention = "upscaling"
                elif use_sd_upscaler:
                    if self.refiner is not None and upscale_step.get("refiner", True):
                        next_intention = "refining"
                    else:
                        next_intention = "inference"

            for it in range(self.iterations):
                if image_callback is not None:
                    def node_image_callback(callback_images: List[PIL.Image.Image]) -> None:
                        """
                        Wrap the original image callback so we're actually pasting the initial image on the main canvas
                        """
                        for j, callback_image in enumerate(callback_images):
                            image_index = (it * self.samples) + j
                            images[image_index].paste(node.resize_image(callback_image), node.bounds[0])
                        image_callback(images)  # type: ignore

                else:
                    node_image_callback = None  # type: ignore

                result = node.execute(
                    pipeline,
                    latent_callback=node_image_callback,
                    next_intention=this_intention if it < self.iterations - 1 else next_intention,
                    use_cached=False,
                    **invocation_kwargs
                )
                
                for j, image in enumerate(result["images"]):
                    image_index = (it * self.samples) + j
                    image = node.resize_image(image)
                    if image.mode == "RGBA":
                        # Draw the alpha mask of the return image onto the outpaint mask
                        alpha = image.split()[-1]
                        black = PIL.Image.new("RGB", alpha.size, (0, 0, 0))
                        outpaint_mask.paste(black, node.bounds[0], mask=alpha)
                        image_draw[image_index].rectangle((*node.bounds[0], *node.bounds[1]), fill=(0, 0, 0, 0))
                        images[image_index].paste(image, node.bounds[0], mask=alpha)
                    else:
                        # Draw a rectangle directly
                        outpaint_draw.rectangle(node.bounds, fill="#000000")
                        images[image_index].paste(node.resize_image(image), node.bounds[0])

                    nsfw_content_detected[image_index] = nsfw_content_detected[image_index] or (
                        "nsfw_content_detected" in result and result["nsfw_content_detected"][j]
                    )

                # Call the callback
                if image_callback is not None:
                    image_callback(images)

        # Determine if there's anything left to outpaint
        image_r_min, image_r_max = outpaint_mask.getextrema()[1]
        if image_r_max > 0 and self.prompt and self.outpaint:
            # Outpaint
            del invocation_kwargs["num_images_per_prompt"]
            outpaint_mask = feather_mask(outpaint_mask)

            outpaint_prompt_tokens = TokenMerger()
            outpaint_prompt_2_tokens = TokenMerger()

            outpaint_negative_prompt_tokens = TokenMerger()
            outpaint_negative_prompt_2_tokens = TokenMerger()

            for i, node in enumerate(self.nodes):
                if node.step.prompt is not None:
                    outpaint_prompt_tokens.add(node.step.prompt)
                if node.step.prompt_2 is not None:
                    outpaint_prompt_2_tokens.add(node.step.prompt_2)
                if node.step.negative_prompt is not None:
                    outpaint_negative_prompt_tokens.add(node.step.negative_prompt)
                if node.step.negative_prompt_2 is not None:
                    outpaint_negative_prompt_2_tokens.add(node.step.negative_prompt_2)

            if self.prompt is not None:
                outpaint_prompt_tokens.add(self.prompt, 2)  # Weighted
            if self.prompt_2 is not None:
                outpaint_prompt_2_tokens.add(self.prompt_2, 2) # Weighted
            if self.negative_prompt is not None:
                outpaint_negative_prompt_tokens.add(self.negative_prompt, 2)
            if self.negative_prompt_2 is not None:
                outpaint_negative_prompt_2_tokens.add(self.negative_prompt_2, 2)

            def outpaint_task_callback(task: str) -> None:
                """
                Wrap the outpaint task callback to include the overall plan task itself
                """
                task_callback(f"Outpaint: {task}")

            invocation_kwargs["task_callback"] = outpaint_task_callback

            for i, image in enumerate(images):
                pipeline.controlnet = None
                if image_callback is not None:
                    def outpaint_image_callback(callback_images: List[PIL.Image.Image]) -> None:
                        """
                        Wrap the original image callback so we're actually pasting the initial image on the main canvas
                        """
                        images[i] = callback_images[0]
                        image_callback(images)  # type: ignore
                else:
                    outpaint_image_callback = None  # type: ignore
                result = pipeline(
                    image=image,
                    mask=outpaint_mask,
                    prompt=str(outpaint_prompt_tokens),
                    prompt_2=str(outpaint_prompt_2_tokens),
                    negative_prompt=str(outpaint_negative_prompt_tokens),
                    negative_prompt_2=str(outpaint_negative_prompt_2_tokens),
                    latent_callback=outpaint_image_callback,
                    num_images_per_prompt=1,
                    **invocation_kwargs,
                )
                images[i] = result["images"][0]
                nsfw_content_detected[i] = nsfw_content_detected[i] or (
                    "nsfw_content_detected" in result and result["nsfw_content_detected"][0]
                )

        return images, nsfw_content_detected

    def get_serialization_dict(self, image_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Serializes the whole plan for storage or passing between processes.
        """
        serialized_image = self.image
        if image_directory is not None and isinstance(self.image, PIL.Image.Image):
            serialized_image = os.path.join(image_directory, f"{get_uuid()}.png")
            self.image.save(serialized_image)

        return {
            "model": self.model,
            "refiner": self.refiner,
            "inpainter": self.inpainter,
            "lora": self.lora,
            "lycoris": self.lycoris,
            "inversion": self.inversion,
            "scheduler": self.scheduler,
            "vae": self.vae,
            "refiner_vae": self.refiner_vae,
            "inpainter_vae": self.inpainter_vae,
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "inpainter_size": self.inpainter_size,
            "refiner_size": self.refiner_size,
            "seed": self.seed,
            "prompt": self.prompt,
            "prompt_2": self.prompt_2,
            "negative_prompt": self.negative_prompt,
            "negative_prompt_2": self.negative_prompt_2,
            "image": serialized_image,
            "nodes": [node.get_serialization_dict(image_directory) for node in self.nodes],
            "samples": self.samples,
            "iterations": self.iterations,
            "upscale_steps": self.upscale_steps,
            "chunking_size": self.chunking_size,
            "chunking_mask_type": self.chunking_mask_type,
            "chunking_mask_kwargs": self.chunking_mask_kwargs,
            "build_tensorrt": self.build_tensorrt,
            "outpaint": self.outpaint,
        }

    @staticmethod
    def deserialize_dict(plan_dict: Dict[str, Any]) -> DiffusionPlan:
        """
        Given a serialized dictionary, instantiate a diffusion plan.
        """
        kwargs = {
            "model": plan_dict["model"],
            "nodes": [DiffusionNode.deserialize_dict(node_dict) for node_dict in plan_dict.get("nodes", [])],
        }

        for arg in [
            "refiner",
            "inpainter",
            "size",
            "refiner_size",
            "inpainter_size",
            "lora",
            "lycoris",
            "inversion",
            "scheduler",
            "vae",
            "refiner_vae",
            "inpainter_vae",
            "width",
            "height",
            "chunking_size",
            "chunking_mask_type",
            "chunking_mask_kwargs",
            "samples",
            "iterations",
            "seed",
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "build_tensorrt",
            "outpaint",
            "upscale_steps"
        ]:
            if arg in plan_dict:
                kwargs[arg] = plan_dict[arg]

        if "image" in plan_dict:
            if isinstance(plan_dict["image"], str) and os.path.exists(plan_dict["image"]):
                kwargs["image"] = PIL.Image.open(plan_dict["image"])
            else:
                kwargs["image"] = plan_dict["image"]

        result = DiffusionPlan(**kwargs)
        return result

    @staticmethod
    def create_mask(width: int, height: int, left: int, top: int, right: int, bottom: int) -> PIL.Image.Image:
        """
        Creates a mask from 6 dimensions
        """
        image = PIL.Image.new("RGB", (width, height))
        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle([(left, top), (right, bottom)], fill="#ffffff")
        return image

    @staticmethod
    def upscale_image(
        image: PIL.Image,
        upscale_steps: Union[UpscaleStepDict, List[UpscaleStepDict]],
        size: Optional[int] = None,
        refiner_size: Optional[int] = None,
        inpainter_size: Optional[int] = None,
        model: Optional[str] = None,
        refiner: Optional[str] = None,
        inpainter: Optional[str] = None,
        lora: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        scheduler: Optional[SCHEDULER_LITERAL] = None,
        vae: Optional[str] = None,
        refiner_vae: Optional[str] = None,
        inpainter_vae: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> DiffusionPlan:
        """
        Generates a plan to upscale a single image
        """
        if kwargs:
            logger.warning(f"Plan `upscale_image` keyword arguments ignored: {kwargs}")
        width, height = image.size
        nodes: List[NodeDict] = [
            {
                "image": image,
                "w": width,
                "h": height,
                "x": 0,
                "y": 0,
            }
        ]
        return DiffusionPlan.assemble(
            size=size,
            refiner_size=refiner_size,
            inpainter_size=inpainter_size,
            model=model,
            refiner=refiner,
            inpainter=inpainter,
            lora=lora,
            lycoris=lycoris,
            inversion=inversion,
            scheduler=scheduler,
            vae=vae,
            refiner_vae=refiner_vae,
            inpainter_vae=inpainter_vae,
            seed=seed,
            width=width,
            height=height,
            upscale_steps=upscale_steps,
            nodes=nodes
        )

    @staticmethod
    def assemble(
        size: Optional[int] = None,
        refiner_size: Optional[int] = None,
        inpainter_size: Optional[int] = None,
        model: Optional[str] = None,
        refiner: Optional[str] = None,
        inpainter: Optional[str] = None,
        lora: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        lycoris: Optional[Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        scheduler: Optional[SCHEDULER_LITERAL] = None,
        vae: Optional[str] = None,
        refiner_vae: Optional[str] = None,
        inpainter_vae: Optional[str] = None,
        model_prompt: Optional[str] = None,
        model_prompt_2: Optional[str] = None,
        model_negative_prompt: Optional[str] = None,
        model_negative_prompt_2: Optional[str] = None,
        samples: int = 1,
        iterations: int = 1,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        nodes: List[NodeDict] = [],
        chunking_size: Optional[int] = None,
        chunking_mask_type: Optional[MASK_TYPE_LITERAL] = None,
        chunking_mask_kwargs: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        num_inference_steps: Optional[int] = DEFAULT_INFERENCE_STEPS,
        mask: Optional[Union[str, PIL.Image.Image]] = None,
        image: Optional[Union[str, PIL.Image.Image]] = None,
        fit: Optional[IMAGE_FIT_LITERAL] = None,
        anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        strength: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
        ip_adapter_image: Optional[Union[str, PIL.Image.Image]] = None,
        ip_adapter_image_fit: Optional[IMAGE_FIT_LITERAL] = None,
        ip_adapter_image_anchor: Optional[IMAGE_ANCHOR_LITERAL] = None,
        control_images: Optional[List[ControlImageDict]] = None,
        remove_background: bool = False,
        fill_background: bool = False,
        scale_to_model_size: bool = False,
        invert_mask: bool = False,
        crop_inpaint: bool = True,
        inpaint_feather: int = 32,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        refiner_start: Optional[float] = DEFAULT_REFINER_START,
        refiner_strength: Optional[float] = DEFAULT_REFINER_STRENGTH,
        refiner_guidance_scale: Optional[float] = DEFAULT_REFINER_GUIDANCE_SCALE,
        refiner_aesthetic_score: Optional[float] = DEFAULT_AESTHETIC_SCORE,
        refiner_negative_aesthetic_score: Optional[float] = DEFAULT_NEGATIVE_AESTHETIC_SCORE,
        refiner_prompt: Optional[str] = None,
        refiner_prompt_2: Optional[str] = None,
        refiner_negative_prompt: Optional[str] = None,
        refiner_negative_prompt_2: Optional[str] = None,
        upscale_steps: Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]] = None,
        **kwargs: Any,
    ) -> DiffusionPlan:
        """
        Assembles a diffusion plan from step dictionaries.
        """
        if kwargs:
            logger.warning(f"Plan `assemble` keyword arguments ignored: {kwargs}")

        # First instantiate the plan
        plan = DiffusionPlan(
            model=model,
            refiner=refiner,
            inpainter=inpainter,
            lora=lora,
            lycoris=lycoris,
            inversion=inversion,
            scheduler=scheduler,
            vae=vae,
            refiner_vae=refiner_vae,
            inpainter_vae=inpainter_vae,
            samples=samples,
            iterations=iterations,
            size=size,
            refiner_size=refiner_size,
            inpainter_size=inpainter_size,
            seed=seed,
            width=width,
            height=height,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            chunking_size=chunking_size,
            chunking_mask_type=chunking_mask_type,
            chunking_mask_kwargs=chunking_mask_kwargs,
            nodes=[],
        )

        # We'll assemble multiple token sets for overall diffusion
        upscale_prompt_tokens = TokenMerger()
        upscale_prompt_2_tokens = TokenMerger()
        upscale_negative_prompt_tokens = TokenMerger()
        upscale_negative_prompt_2_tokens = TokenMerger()

        # Helper method for getting the upscale list with merged prompts
        def get_upscale_steps() -> Optional[Union[UpscaleStepDict, List[UpscaleStepDict]]]:
            if upscale_steps is None:
                return None
            elif isinstance(upscale_steps, list):
                return [
                    {
                        **step, # type: ignore[misc]
                        **{
                            "prompt": str(
                                upscale_prompt_tokens.clone(step.get("prompt", None))
                            ),
                            "prompt_2": str(
                                upscale_prompt_2_tokens.clone(step.get("prompt_2", None))
                            ),
                            "negative_prompt": str(
                                upscale_negative_prompt_tokens.clone(step.get("negative_prompt", None))
                            ),
                            "negative_prompt_2": str(
                                upscale_negative_prompt_2_tokens.clone(step.get("negative_prompt_2", None))
                            ),
                        }
                    }
                    for step in upscale_steps
                ]
            else:
                return { # type: ignore[return-value]
                    **upscale_steps, # type: ignore[misc]
                    **{
                        "prompt": str(
                            upscale_prompt_tokens.clone(upscale_steps.get("prompt", None))
                        ),
                        "prompt_2": str(
                            upscale_prompt_2_tokens.clone(upscale_steps.get("prompt_2", None))
                        ),
                        "negative_prompt": str(
                            upscale_negative_prompt_tokens.clone(upscale_steps.get("negative_prompt", None))
                        ),
                        "negative_prompt_2": str(
                            upscale_negative_prompt_2_tokens.clone(upscale_steps.get("negative_prompt_2", None))
                        ),
                    }
                }

        refiner_prompt_tokens = TokenMerger()
        refiner_prompt_2_tokens = TokenMerger()
        refiner_negative_prompt_tokens = TokenMerger()
        refiner_negative_prompt_2_tokens = TokenMerger()

        if prompt:
            upscale_prompt_tokens.add(prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if prompt_2:
            upscale_prompt_2_tokens.add(prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if negative_prompt:
            upscale_negative_prompt_tokens.add(negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if negative_prompt_2:
            upscale_negative_prompt_2_tokens.add(negative_prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT)

        if model_prompt:
            refiner_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)
            upscale_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)
        if model_prompt_2:
            refiner_prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)
            upscale_prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)

        if model_negative_prompt:
            refiner_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
            upscale_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
        if model_negative_prompt_2:
            refiner_negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
            upscale_negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)

        if refiner_prompt:
            refiner_prompt_tokens.add(refiner_prompt)
            refiner_prompt = str(refiner_prompt_tokens)
        else:
            refiner_prompt = None
        
        if refiner_prompt_2:
            refiner_prompt_2_tokens.add(refiner_prompt_2)
            refiner_prompt_2 = str(refiner_prompt_2_tokens)
        else:
            refiner_prompt_2 = None
        
        if refiner_negative_prompt:
            refiner_negative_prompt_tokens.add(refiner_negative_prompt)
            refiner_negative_prompt = str(refiner_negative_prompt_tokens)
        else:
            refiner_negative_prompt = None
        
        if refiner_negative_prompt_2:
            refiner_negative_prompt_2_tokens.add(refiner_negative_prompt_2)
            refiner_negative_prompt_2 = str(refiner_negative_prompt_2_tokens)
        else:
            refiner_negative_prompt_2 = None

        # Now assemble the diffusion steps
        node_count = len(nodes)

        if node_count == 0:
            # No nodes/canvas, create a plan from one given step
            name = "Text to Image"
            prompt_tokens = TokenMerger()
            if prompt:
                prompt_tokens.add(prompt)
            if model_prompt:
                prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)
            
            prompt_2_tokens = TokenMerger()
            if prompt_2:
                prompt_2_tokens.add(prompt_2)
            if model_prompt_2:
                prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)

            negative_prompt_tokens = TokenMerger()
            if negative_prompt:
                negative_prompt_tokens.add(negative_prompt)
            if model_negative_prompt:
                negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)

            negative_prompt_2_tokens = TokenMerger()
            if negative_prompt_2:
                negative_prompt_2_tokens.add(negative_prompt_2)
            if model_negative_prompt_2:
                negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)

            if image:
                if isinstance(image, str):
                    image = PIL.Image.open(image)
                if width and height:
                    image = fit_image(image, width, height, fit, anchor)
                else:
                    width, height = image.size
                
            if mask:
                if isinstance(mask, str):
                    mask = PIL.Image.open(mask)
                if width and height:
                    mask = fit_image(mask, width, height, fit, anchor)
                else:
                    width, height = mask.size
            
            if ip_adapter_image:
                if isinstance(ip_adapter_image, str):
                    ip_adapter_image = PIL.Image.open(ip_adapter_image)
                if width and height:
                    ip_adapter_image = fit_image(ip_adapter_image, width, height, ip_adapter_image_fit, ip_adapter_image_anchor)
                else:
                    width, height = ip_adapter_image.size

            if control_images:
                for i, control_image_dict in enumerate(control_images):
                    control_image = control_image_dict["image"]
                    control_anchor = control_image_dict.get("anchor", anchor)
                    control_fit = control_image_dict.get("fit", fit)

                    if isinstance(control_image, str):
                        control_image = PIL.Image.open(control_image)
                    if width and height:
                        control_image = fit_image(control_image, width, height, fit, anchor)
                    else:
                        width, height = control_image.size

                    control_images[i] = { # type: ignore[call-overload]
                        "image": control_image,
                        "controlnet": control_image_dict["controlnet"],
                        "process": control_image_dict.get("process", True),
                        "scale": control_image_dict.get("scale", 0.5),
                        "invert": control_image_dict.get("invert", False)
                    }

            if mask and invert_mask:
                mask = PIL.ImageOps.invert(mask.convert("L"))
            
            if image and remove_background and fill_background:
                image = execute_remove_background(image)
                remove_background = False
                
                if not mask:
                    mask = PIL.Image.new("RGB", image.size, (255, 255, 255))

                alpha = image.split()[-1]
                black = PIL.Image.new("RGB", image.size, (0, 0, 0))
                mask.paste(black, mask=alpha)

            if control_images and image and mask and ip_adapter_scale:
                name = "Controlled Inpainting with Image Prompting"
            elif control_images and image and mask:
                name = "Controlled Inpainting"
            elif control_images and image and ip_adapter_scale and strength:
                name = "Controlled Image to Image with Image Prompting"
            elif control_images and image and ip_adapter_scale:
                name = "Controlled Text to Image with Image Prompting"
            elif control_images and image and strength:
                name = "Controlled Image to Image"
            elif control_images:
                name = "Controlled Text to Image"
            elif image and mask and ip_adapter_scale:
                name = "Inpainting with Image Prompting"
            elif image and mask:
                name = "Inpainting"
            elif image and strength and ip_adapter_scale:
                name = "Image to Image with Image Prompting"
            elif image and ip_adapter_scale:
                name = "Text to Image with Image Prompting"
            elif image and strength:
                name = "Image to Image"
            else:
                name = "Text to Image"

            step = DiffusionStep(
                name=name,
                width=width,
                height=height,
                prompt=str(prompt_tokens),
                prompt_2=str(prompt_2_tokens),
                negative_prompt=str(negative_prompt_tokens),
                negative_prompt_2=str(negative_prompt_2_tokens),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                ip_adapter_scale=ip_adapter_scale,
                ip_adapter_image=ip_adapter_image,
                refiner_start=refiner_start,
                refiner_strength=refiner_strength,
                refiner_guidance_scale=refiner_guidance_scale,
                refiner_aesthetic_score=refiner_aesthetic_score,
                refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                refiner_prompt=refiner_prompt,
                refiner_prompt_2=refiner_prompt_2,
                refiner_negative_prompt=refiner_negative_prompt,
                refiner_negative_prompt_2=refiner_negative_prompt_2,
                crop_inpaint=crop_inpaint,
                inpaint_feather=inpaint_feather,
                image=image,
                mask=mask,
                remove_background=remove_background,
                scale_to_model_size=scale_to_model_size,
                control_images=control_images
            )

            if not width:
                width = plan.width # Default
            if not height:
                height = plan.height # Default
            
            # Change plan defaults if passed
            plan.width = width
            plan.height = height
            
            # Assemble node
            plan.nodes = [DiffusionNode([(0, 0), (width, height)], step)]
            plan.upscale_steps = get_upscale_steps()
            return plan

        # Using the diffusion canvas, assemble a multi-step plan
        for i, node_dict in enumerate(nodes):
            step = DiffusionStep(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                refiner_start=refiner_start,
                refiner_strength=refiner_strength,
                refiner_guidance_scale=refiner_guidance_scale,
                refiner_aesthetic_score=refiner_aesthetic_score,
                refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                refiner_prompt=refiner_prompt,
                refiner_prompt_2=refiner_prompt_2,
                refiner_negative_prompt=refiner_negative_prompt,
                refiner_negative_prompt_2=refiner_negative_prompt_2,
            )

            node_left = int(node_dict.get("x", 0))
            node_top = int(node_dict.get("y", 0))
            node_fit = node_dict.get("fit", None)
            node_anchor = node_dict.get("anchor", None)

            node_prompt = node_dict.get("prompt", None)
            node_prompt_2 = node_dict.get("prompt_2", None)
            node_negative_prompt = node_dict.get("negative_prompt", None)
            node_negative_prompt_2 = node_dict.get("negative_prompt_2", None)
            node_strength: Optional[float] = node_dict.get("strength", None)
            node_image = node_dict.get("image", None)
            node_inpaint_mask = node_dict.get("mask", None)
            node_crop_inpaint = node_dict.get("crop_inpaint", crop_inpaint)
            node_inpaint_feather = node_dict.get("inpaint_feather", inpaint_feather)
            node_invert_mask = node_dict.get("invert_mask", False)
            node_scale_to_model_size = bool(node_dict.get("scale_to_model_size", False))
            node_remove_background = bool(node_dict.get("remove_background", False))
            node_ip_adapter_scale: Optional[float] = node_dict.get("ip_adapter_scale", None) # type: ignore[assignment]
            node_ip_adapter_image: Optional[float] = node_dict.get("ip_adapter_image", None) # type: ignore[assignment]
            node_ip_adapter_image_fit: Optional[str] = node_dict.get("ip_adapter_image_fit", None) # type: ignore[assignment]
            node_ip_adapter_image_anchor: Optional[str] = node_dict.get("ip_adapter_image_anchor", None) # type: ignore[assignment]

            node_control_images: Optional[List[ControlImageDict]] = node_dict.get("control_images", None)
            node_inference_steps: Optional[int] = node_dict.get("inference_steps", None)  # type: ignore[assignment]
            node_guidance_scale: Optional[float] = node_dict.get("guidance_scale", None)  # type: ignore[assignment]
            node_refiner_start: Optional[float] = node_dict.get("refiner_start", None) # type: ignore[assignment]
            node_refiner_strength: Optional[float] = node_dict.get("refiner_strength", None)  # type: ignore[assignment]
            node_refiner_guidance_scale: Optional[float] = node_dict.get("refiner_guidance_scale", None)  # type: ignore[assignment]
            node_refiner_aesthetic_score: Optional[float] = node_dict.get("refiner_aesthetic_score", None)  # type: ignore[assignment]
            node_refiner_negative_aesthetic_score: Optional[float] = node_dict.get("refiner_negative_aesthetic_score", None)  # type: ignore[assignment]

            node_prompt_tokens = TokenMerger()
            node_prompt_2_tokens = TokenMerger()
            node_negative_prompt_tokens = TokenMerger()
            node_negative_prompt_2_tokens = TokenMerger()

            if "w" in node_dict:
                node_width = int(node_dict["w"])
            elif node_image is not None:  # type: ignore[unreachable]
                node_width, _ = node_image.size
            elif node_inpaint_mask is not None:
                node_width, _ = node_inpaint_mask.size
            elif node_ip_adapter_image is not None:
                node_width, _ = node_ip_adapter_image.size
            elif node_control_images:
                node_width, _ = node_control_images[next(iter(node_control_images))][0]["image"].size
            else:
                raise ValueError(f"Node {i} missing width, pass 'w' or an image")
            if "h" in node_dict:
                node_height = int(node_dict["h"])
            elif node_image is not None:  # type: ignore[unreachable]
                _, node_height = node_image.size
            elif node_inpaint_mask is not None:
                _, node_height = node_inpaint_mask.size
            elif node_ip_adapter_image is not None:
                _, node_height = node_ip_adapter_image.size
            elif node_control_images:
                _, node_height = node_control_images[next(iter(node_control_images))][0]["image"].size
            else:
                raise ValueError(f"Node {i} missing height, pass 'h' or an image")

            node_bounds = [
                (node_left, node_top),
                (node_left + node_width, node_top + node_height),
            ]

            if node_prompt:
                node_prompt_tokens.add(node_prompt)
                upscale_prompt_tokens.add(node_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if prompt and (node_image or node_ip_adapter_image or node_control_images):
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_prompt_tokens.add(prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_prompt:
                node_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)

            if node_prompt_2:
                node_prompt_2_tokens.add(node_prompt_2)
                upscale_prompt_2_tokens.add(node_prompt_2, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if prompt_2 and (node_image or node_ip_adapter_image or node_control_images):
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_prompt_2_tokens.add(prompt_2, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_prompt_2:
                node_prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)

            if node_negative_prompt:
                node_negative_prompt_tokens.add(node_negative_prompt)
                upscale_negative_prompt_tokens.add(node_negative_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if negative_prompt and (node_image or node_ip_adapter_image or node_control_images):
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_negative_prompt_tokens.add(negative_prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_negative_prompt:
                node_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
            
            if node_negative_prompt_2:
                node_negative_prompt_tokens.add(node_negative_prompt_2)
                upscale_negative_prompt_2_tokens.add(node_negative_prompt_2, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if negative_prompt_2 and (node_image or node_ip_adapter_image or node_control_images):
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_negative_prompt_2_tokens.add(negative_prompt_2, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_negative_prompt_2:
                node_negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)

            black = PIL.Image.new("RGB", (node_width, node_height), (0, 0, 0))
            white = PIL.Image.new("RGB", (node_width, node_height), (255, 255, 255))
            outpaint_steps: List[DiffusionStep] = []
            outpainted_images: Dict[int, PIL.Image.Image] = {}

            def prepare_image(
                image: PIL.Image.Image,
                outpaint_if_necessary: bool = False,
                mask: Optional[PIL.Image.Image] = None,
                fit: Optional[IMAGE_FIT_LITERAL] = None,
                anchor: Optional[IMAGE_ANCHOR_LITERAL] = None
            )-> Union[Tuple[PIL.Image.Image, PIL.Image.Image], Tuple[DiffusionStep, Any]]:
                """
                Checks if the image needs to be outpainted
                """
                nonlocal outpainted_images
                for step_index, outpainted_image in outpainted_images.items():
                    if images_are_equal(outpainted_image, image):
                        return outpaint_steps[step_index], 0

                image_mask = PIL.Image.new("RGB", (node_width, node_height), (255, 255, 255)) # Mask for outpainting if needed
                fitted_image = fit_image(image, node_width, node_height, fit, anchor)
                fitted_alpha = fitted_image.split()[-1]
                fitted_alpha_clamp = PIL.Image.eval(fitted_alpha, lambda a: 255 if a > 128 else 0)

                image_mask.paste(black, mask=fitted_alpha)

                if mask:
                    image_mask.paste(mask)
                    fitted_inverse_alpha = PIL.Image.eval(fitted_alpha_clamp, lambda a: 255 - a)
                    image_mask.paste(white, mask=fitted_inverse_alpha)

                image_mask_r_min, image_mask_r_max = image_mask.getextrema()[1]
                image_needs_outpainting = image_mask_r_max > 0

                if image_needs_outpainting and outpaint_if_necessary:
                    step = DiffusionStep(
                        name=f"Outpaint Node {i+1}",
                        image=fitted_image,
                        mask=feather_mask(image_mask.convert("1")),
                        prompt=str(node_prompt_tokens),
                        prompt_2=str(node_prompt_2_tokens),
                        negative_prompt=str(node_negative_prompt_tokens),
                        negative_prompt_2=str(node_negative_prompt_2_tokens),
                        guidance_scale=guidance_scale,
                        num_inference_steps=node_inference_steps if node_inference_steps else num_inference_steps,
                        crop_inpaint=node_crop_inpaint,
                        inpaint_feather=node_inpaint_feather,
                        refiner_start=refiner_start,
                        refiner_strength=refiner_strength,
                        refiner_guidance_scale=refiner_guidance_scale,
                        refiner_aesthetic_score=refiner_aesthetic_score,
                        refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                        refiner_prompt=refiner_prompt,
                        refiner_prompt_2=refiner_prompt_2,
                        refiner_negative_prompt=refiner_negative_prompt,
                        refiner_negative_prompt_2=refiner_negative_prompt_2,
                    )
                    outpaint_steps.append(step)
                    outpainted_images[len(outpaint_steps)-1] = image
                    return step, None
                return fitted_image, image_mask

            will_infer = node_strength is not None or node_inpaint_mask is not None

            if node_inpaint_mask:
                node_inpaint_mask = node_inpaint_mask.convert("L")
                if node_invert_mask:
                    node_inpaint_mask = PIL.ImageOps.invert(node_inpaint_mask)

            if node_image:
                if node_ip_adapter_scale and not node_ip_adapter_image:
                    node_ip_adapter_image = node_image
                if node_remove_background and will_infer:
                    node_remove_background = False # Don't double-remove
                    node_image, new_inpaint_mask = prepare_image(
                        execute_remove_background(node_image),
                        mask=node_inpaint_mask,
                        fit=node_fit,
                        anchor=node_anchor
                    )
                    if node_inpaint_mask:
                        node_inpaint_mask = new_inpaint_mask
                else:
                    node_image, new_inpaint_mask = prepare_image(
                        node_image,
                        mask=node_inpaint_mask,
                        fit=node_fit,
                        anchor=node_anchor
                    )
                    if node_inpaint_mask:
                        node_inpaint_mask = new_inpaint_mask

            if node_ip_adapter_image:
                node_ip_adapter_image = prepare_image( # type: ignore[assignment]
                    node_ip_adapter_image,
                    fit=node_ip_adapter_image_fit, # type: ignore[arg-type]
                    anchor=node_ip_adapter_image_anchor, # type: ignore[arg-type]
                )[0]

            if node_control_images:
                node_control_images = [
                    {
                        "image": prepare_image(
                            control_image["image"],
                            fit=control_image.get("fit", None),
                            anchor=control_image.get("anchor", None),
                            outpaint_if_necessary=True
                        )[0],
                        "controlnet": control_image["controlnet"],
                        "scale": control_image.get("scale", 1.0),
                        "process": control_image.get("process", True),
                        "invert": control_image.get("invert", False),
                    }
                    for control_image in node_control_images
                ]

            node_prompt_str = str(node_prompt_tokens)
            node_prompt_2_str = str(node_prompt_2_tokens)
            node_negative_prompt_str = str(node_negative_prompt_tokens)
            node_negative_prompt_2_str = str(node_negative_prompt_2_tokens)

            if node_inpaint_mask:
                node_inpaint_mask_r_min, node_inpaint_mask_r_max = node_inpaint_mask.getextrema()[1]
                image_needs_inpainting = node_inpaint_mask_r_max > 0
            else:
                image_needs_inpainting = False

            if node_strength is None or not image_needs_inpainting:
                node_inpaint_mask = None

            if node_control_images and node_image and node_inpaint_mask and node_ip_adapter_scale:
                name = "Controlled Inpainting with Image Prompting"
            elif node_control_images and node_image and node_inpaint_mask:
                name = "Controlled Inpainting"
            elif node_control_images and node_image and node_ip_adapter_scale and node_strength:
                name = "Controlled Image to Image with Image Prompting"
            elif node_control_images and (node_image or node_ip_adapter_image) and node_ip_adapter_scale:
                name = "Controlled Text to Image with Image Prompting"
            elif node_control_images and node_image and node_strength:
                name = "Controlled Image to Image"
            elif node_control_images:
                name = "Controlled Text to Image"
            elif node_image and node_inpaint_mask and node_ip_adapter_scale:
                name = "Inpainting with Image Prompting"
            elif node_image and node_inpaint_mask:
                name = "Inpainting"
            elif node_image and node_strength and node_ip_adapter_scale:
                name = "Image to Image with Image Prompting"
            elif (node_image or node_ip_adapter_image) and node_ip_adapter_scale:
                name = "Text to Image with Image Prompting"
            elif node_image and node_strength:
                name = "Image to Image"
            elif node_image:
                name = "Image Pass-Through"
                if node_width == width and node_height == height:
                    plan.outpaint = False
                node_prompt_str = None # type: ignore[assignment]
                node_prompt_2_str = None # type: ignore[assignment]
                node_negative_prompt_str = None # type: ignore[assignment]
                node_negative_prompt_2_str = None # type: ignore[assignment]
            else:
                name = "Text to Image"

            step = DiffusionStep(
                name=f"{name} Node {i+1}",
                image=node_image,
                mask=node_inpaint_mask,
                prompt=node_prompt_str,
                prompt_2=node_prompt_2_str,
                negative_prompt=node_negative_prompt_str,
                negative_prompt_2=node_negative_prompt_2_str,
                crop_inpaint=node_crop_inpaint,
                inpaint_feather=node_inpaint_feather,
                strength=node_strength,
                guidance_scale=guidance_scale,
                num_inference_steps=node_inference_steps if node_inference_steps else num_inference_steps,
                ip_adapter_image=node_ip_adapter_image,
                ip_adapter_scale=node_ip_adapter_scale,
                control_images=node_control_images,
                refiner_start=refiner_start,
                refiner_strength=refiner_strength,
                refiner_guidance_scale=refiner_guidance_scale,
                refiner_aesthetic_score=refiner_aesthetic_score,
                refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                refiner_prompt=refiner_prompt,
                refiner_prompt_2=refiner_prompt_2,
                refiner_negative_prompt=refiner_negative_prompt,
                refiner_negative_prompt_2=refiner_negative_prompt_2,
                remove_background=node_remove_background,
                scale_to_model_size=node_scale_to_model_size
            )

            # Add step to plan
            plan.nodes.append(DiffusionNode(node_bounds, step))
        plan.upscale_steps = get_upscale_steps()
        return plan
