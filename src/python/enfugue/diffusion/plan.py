from __future__ import annotations

import io
import os
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import math

from PIL.PngImagePlugin import PngInfo

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    List,
    Callable,
    TypedDict,
    TYPE_CHECKING,
)

from pibble.util.strings import get_uuid, Serializer

from enfugue.util import (
    logger,
    feather_mask,
    fit_image,
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
    )

DEFAULT_SIZE = 512
DEFAULT_IMAGE_CALLBACK_STEPS = 10
DEFAULT_CONDITIONING_SCALE = 1.0
DEFAULT_IMG2IMG_STRENGTH = 0.8
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_UPSCALE_PROMPT = "highly detailed, ultra-detailed, intricate detail, high definition, HD, 4k, 8k UHD"
DEFAULT_UPSCALE_NEGATIVE_PROMPT = ""
DEFAULT_UPSCALE_DIFFUSION_STEPS = 100
DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE = 12
DEFAULT_UPSCALE_DIFFUSION_STRENGTH = 0.2

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


class NodeDict(TypedDict):
    w: int
    h: int
    x: int
    y: int
    fit: Optional[IMAGE_FIT_LITERAL]
    anchor: Optional[IMAGE_ANCHOR_LITERAL]
    infer: Optional[bool]
    inpaint: Optional[bool]
    control: Optional[bool]
    controlnet: Optional[str]
    prompt: Optional[str]
    prompt_2: Optional[str]
    negative_prompt: Optional[str]
    negative_prompt_2: Optional[str]
    strength: Optional[float]
    conditioning_scale: Optional[float]
    image: Optional[PIL.Image.Image]
    mask: Optional[PIL.Image.Image]
    process_control_image: Optional[bool]
    remove_background: Optional[bool]
    invert: Optional[bool]
    invert_mask: Optional[bool]
    crop_inpaint: Optional[bool]
    inpaint_feather: Optional[int]


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
        control_image: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        controlnet: Optional[CONTROLNET_LITERAL] = None,
        conditioning_scale: Optional[float] = DEFAULT_CONDITIONING_SCALE,
        strength: Optional[float] = DEFAULT_IMG2IMG_STRENGTH,
        num_inference_steps: Optional[int] = DEFAULT_INFERENCE_STEPS,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        refiner_strength: Optional[float] = DEFAULT_REFINER_STRENGTH,
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
        process_control_image: bool = True,
        scale_to_model_size: bool = True,
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
        self.control_image = control_image
        self.controlnet = controlnet
        self.strength = strength if strength is not None else DEFAULT_IMG2IMG_STRENGTH
        self.conditioning_scale = conditioning_scale if conditioning_scale is not None else DEFAULT_CONDITIONING_SCALE
        self.num_inference_steps = num_inference_steps if num_inference_steps is not None else DEFAULT_INFERENCE_STEPS
        self.guidance_scale = guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE
        self.refiner_strength = refiner_strength if refiner_strength is not None else DEFAULT_REFINER_STRENGTH
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
        self.refiner_prompt = refiner_prompt
        self.refiner_prompt_2 = refiner_prompt_2
        self.refiner_negative_prompt = refiner_negative_prompt
        self.refiner_negative_prompt_2 = refiner_negative_prompt_2
        self.remove_background = remove_background
        self.process_control_image = process_control_image
        self.scale_to_model_size = scale_to_model_size
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
            "controlnet": self.controlnet,
            "conditioning_scale": self.conditioning_scale,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "remove_background": self.remove_background,
            "refiner_strength": self.refiner_strength,
            "refiner_guidance_scale": self.refiner_guidance_scale,
            "refiner_aesthetic_score": self.refiner_aesthetic_score,
            "refiner_negative_aesthetic_score": self.refiner_negative_aesthetic_score,
            "refiner_prompt": self.refiner_prompt,
            "refiner_prompt_2": self.refiner_prompt_2,
            "refiner_negative_prompt": self.refiner_negative_prompt,
            "refiner_negative_prompt_2": self.refiner_negative_prompt_2,
            "process_control_image": self.process_control_image,
            "scale_to_model_size": self.scale_to_model_size,
            "crop_inpaint": self.crop_inpaint,
            "inpaint_feather": self.inpaint_feather,
        }

        serialize_children: List[DiffusionStep] = []
        for key in ["image", "mask", "control_image"]:
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
            "control_image": self.control_image,
            "conditioning_scale": self.conditioning_scale,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
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

    def check_process_control_image(
        self, pipeline: DiffusionPipelineManager, control_image: Optional[PIL.Image.Image]
    ) -> Optional[PIL.Image.Image]:
        """
        Gets the control image for the pipeline based on the requested controlnet
        """
        if self.controlnet is None or control_image is None:
            return None
        if self.process_control_image:
            if self.controlnet == "canny":
                return pipeline.edge_detector.canny(control_image)
            if self.controlnet == "hed":
                return pipeline.edge_detector.hed(control_image)
            if self.controlnet == "scribble":
                return pipeline.edge_detector.hed(control_image, scribble=True)
            if self.controlnet == "pidi":
                return pipeline.edge_detector.pidi(control_image)
            if self.controlnet == "depth":
                return pipeline.depth_detector.midas(control_image)
            if self.controlnet == "normal":
                return pipeline.depth_detector.normal(control_image)
            if self.controlnet == "pose":
                return pipeline.pose_detector.detect(control_image)
            if self.controlnet == "line":
                return pipeline.line_detector.detect(control_image)
            if self.controlnet == "anime":
                return pipeline.line_detector.detect(control_image, anime=True)
            if self.controlnet == "mlsd":
                return pipeline.line_detector.mlsd(control_image)
        return control_image

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

        if isinstance(self.mask, DiffusionStep):
            mask = self.mask.execute(pipeline, samples=1, **kwargs)["images"][0]
        elif isinstance(self.mask, str):
            mask = PIL.Image.open(self.mask)
        else:
            mask = self.mask

        if isinstance(self.control_image, DiffusionStep):
            control_image = self.control_image.execute(pipeline, samples=1, **kwargs)["images"][0]
        elif isinstance(self.control_image, str):
            control_image = PIL.Image.open(self.control_image)
        else:
            control_image = self.control_image

        if not self.prompt:
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

        pipeline.controlnet = self.controlnet
        invocation_kwargs = {**kwargs, **self.kwargs}

        image_scale = 1
        pipeline_size = pipeline.inpainter_size if mask is not None else pipeline.size
        image_width, image_height, image_background, image_position = None, None, None, None

        if image is not None:
            image_width, image_height = image.size
            invocation_kwargs["image"] = image
        if control_image is not None:
            if image is not None:
                assert image.size == control_image.size
            else:
                image_width, image_height = control_image.size
            invocation_kwargs["control_image"] = control_image
        if mask is not None:
            mask_width, mask_height = mask.size
            if (
                self.crop_inpaint
                and (mask_width > pipeline_size or mask_height > pipeline.size)
                and image is not None
                and control_image is None
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
            if control_image is not None:
                assert control_image.size == mask.size
            else:
                image_width, image_height = mask.size

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
        invocation_kwargs["control_image"] = self.check_process_control_image(pipeline, control_image)

        if image_scale > 1:
            # scale input images up
            for key in ["image", "mask", "control_image"]:
                if invocation_kwargs.get(key, None) is not None:
                    invocation_kwargs[key] = self.scale_image(invocation_kwargs[key], image_scale)

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
            "controlnet",
            "conditioning_scale",
            "strength",
            "num_inference_steps",
            "guidance_scale",
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
            "process_control_image",
            "scale_to_model_size",
            "crop_inpaint",
            "inpaint_feather"
        ]:
            if key in step_dict:
                kwargs[key] = step_dict[key]

        deserialized_children = [DiffusionStep.deserialize_dict(child) for child in step_dict.get("children", [])]
        for key in ["image", "control_image", "mask"]:
            if key not in step_dict:
                continue
            if isinstance(step_dict[key], int):
                kwargs[key] = deserialized_children[step_dict[key]]

            elif isinstance(step_dict[key], str) and os.path.exists(step_dict[key]):
                kwargs[key] = PIL.Image.open(step_dict[key])
            else:
                kwargs[key] = step_dict[key]

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
        chunking_blur: Optional[int] = None,
        samples: Optional[int] = 1,
        iterations: Optional[int] = 1,
        seed: Optional[int] = None,
        build_tensorrt: bool = False,
        outscale: Optional[int] = 1,
        upscale: Optional[Union[UPSCALE_LITERAL, List[UPSCALE_LITERAL]]] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_pipeline: Optional[str] = None,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = DEFAULT_UPSCALE_DIFFUSION_STEPS,
        upscale_diffusion_guidance_scale: Optional[
            Union[float, int, List[Union[float, int]]]
        ] = DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE,
        upscale_diffusion_strength: Optional[Union[float, List[float]]] = DEFAULT_UPSCALE_DIFFUSION_STRENGTH,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_PROMPT,
        upscale_diffusion_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_NEGATIVE_PROMPT,
        upscale_diffusion_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_controlnet: Optional[Union[CONTROLNET_LITERAL, List[CONTROLNET_LITERAL]]] = None,
        upscale_diffusion_chunking_size: Optional[int] = None,
        upscale_diffusion_chunking_blur: Optional[int] = None,
        upscale_diffusion_scale_chunking_size: bool = True,
        upscale_diffusion_scale_chunking_blur: bool = True,
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
        self.chunking_blur = chunking_blur if chunking_blur is not None else self.size // 8  # Pass 0 to disable
        self.samples = samples if samples is not None else 1
        self.iterations = iterations if iterations is not None else 1
        self.seed = seed

        self.build_tensorrt = build_tensorrt
        self.nodes = nodes
        self.outscale = outscale if outscale is not None else 1
        self.upscale = upscale
        self.upscale_iterative = bool(upscale_iterative)
        self.upscale_diffusion = bool(upscale_diffusion)
        self.upscale_pipeline = upscale_pipeline
        self.upscale_diffusion_chunking_size = (
            upscale_diffusion_chunking_size if upscale_diffusion_chunking_size is not None else self.size // 4
        )
        self.upscale_diffusion_chunking_blur = (
            upscale_diffusion_chunking_blur if upscale_diffusion_chunking_blur is not None else self.size // 4
        )
        self.upscale_diffusion_guidance_scale = (
            upscale_diffusion_guidance_scale
            if upscale_diffusion_guidance_scale is not None
            else DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE
        )
        self.upscale_diffusion_steps = (
            upscale_diffusion_steps if upscale_diffusion_steps is not None else DEFAULT_UPSCALE_DIFFUSION_STEPS
        )
        self.upscale_diffusion_strength = (
            upscale_diffusion_strength if upscale_diffusion_strength is not None else DEFAULT_UPSCALE_DIFFUSION_STRENGTH
        )
        self.upscale_diffusion_prompt = (
            upscale_diffusion_prompt if upscale_diffusion_prompt is not None else DEFAULT_UPSCALE_PROMPT
        )
        self.upscale_diffusion_prompt_2 = upscale_diffusion_prompt_2
        self.upscale_diffusion_negative_prompt = (
            upscale_diffusion_negative_prompt
            if upscale_diffusion_negative_prompt is not None
            else DEFAULT_UPSCALE_NEGATIVE_PROMPT
        )
        self.upscale_diffusion_negative_prompt_2 = upscale_diffusion_negative_prompt_2
        self.upscale_diffusion_controlnet = upscale_diffusion_controlnet
        self.upscale_diffusion_scale_chunking_size = bool(upscale_diffusion_scale_chunking_size)
        self.upscale_diffusion_scale_chunking_blur = bool(upscale_diffusion_scale_chunking_blur)

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the keyword arguments that will be passing to the pipeline call.
        """
        return {
            "width": self.width,
            "height": self.height,
            "chunking_size": self.chunking_size,
            "chunking_blur": self.chunking_blur,
            "num_images_per_prompt": self.samples,
        }

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

        if self.upscale is not None and self.outscale > 1:
            if self.upscale_iterative:
                scales = [2] * int(math.log(self.outscale, 2))
            else:
                scales = [self.outscale]
            for j, scale in enumerate(scales):

                def get_item_for_scale(item: Any) -> Any:
                    if not isinstance(item, list):
                        return item
                    elif len(item) < j + 1:
                        if len(item) == 0:
                            return None
                        return item[-1]
                    else:
                        return item[j]

                for i, image in enumerate(images):
                    if nsfw is not None and nsfw[i]:
                        logger.debug(f"Image {i} had NSFW content, not upscaling.")
                        continue

                    upscale = get_item_for_scale(self.upscale).lower()
                    logger.debug(f"Upscaling sample {i} by {scale} using {upscale}")
                    task_callback(f"Upscaling sample {i+1}")

                    if upscale in ["esrgan", "esrganime", "gfpgan"]:
                        if self.refiner:
                            pipeline.unload_pipeline("clearing memory for upscaler")
                            pipeline.offload_refiner()
                        else:
                            pipeline.offload_pipeline()
                            pipeline.unload_refiner("clearing memory for upscaler")

                    if upscale == "esrgan":
                        image = pipeline.upscaler.esrgan(image, tile=pipeline.size, outscale=scale)
                    elif upscale == "esrganime":
                        image = pipeline.upscaler.esrgan(image, tile=pipeline.size, outscale=scale, anime=True)
                    elif upscale == "gfpgan":
                        image = pipeline.upscaler.gfpgan(image, tile=pipeline.size, outscale=scale)
                    elif upscale in PIL_INTERPOLATION:
                        width, height = image.size
                        image = image.resize((width * scale, height * scale), resample=PIL_INTERPOLATION[upscale])
                    else:
                        logger.error(f"Unknown upscaler {upscale}")
                        return images

                    images[i] = image
                    if image_callback is not None:
                        image_callback(images)

                if self.upscale_diffusion:
                    task_callback("Preparing upscale pipeline")
                    if self.refiner and self.upscale_pipeline != "base":
                        # Refiners have safety disabled from the jump
                        logger.debug("Using refiner for upscaling.")
                        re_enable_safety = False
                        pipeline.reload_refiner()
                    else:
                        # Disable pipeline safety here, it gives many false positives when upscaling.
                        # We'll re-enable it after.
                        logger.debug("Using base pipeline for upscaling.")
                        re_enable_safety = pipeline.safe
                        pipeline.safe = False
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
                            "prompt": get_item_for_scale(self.upscale_diffusion_prompt),
                            "prompt_2": get_item_for_scale(self.upscale_diffusion_prompt_2),
                            "negative_prompt": get_item_for_scale(self.upscale_diffusion_negative_prompt),
                            "negative_prompt_2": get_item_for_scale(self.upscale_diffusion_negative_prompt_2),
                            "strength": get_item_for_scale(self.upscale_diffusion_strength),
                            "num_inference_steps": get_item_for_scale(self.upscale_diffusion_steps),
                            "guidance_scale": get_item_for_scale(self.upscale_diffusion_guidance_scale),
                            "chunking_size": self.upscale_diffusion_chunking_size,
                            "chunking_blur": self.upscale_diffusion_chunking_blur,
                            "progress_callback": progress_callback,
                        }

                        upscale_controlnet = get_item_for_scale(self.upscale_diffusion_controlnet)
                        if upscale_controlnet is not None:
                            logger.debug(f"Enabling {upscale_controlnet} for upscale diffusion")
                            if upscale_controlnet == "canny":
                                pipeline.controlnet = "canny"
                                kwargs["control_image"] = pipeline.edge_detector.canny(image)
                            elif upscale_controlnet == "hed":
                                pipeline.controlnet = "hed"
                                kwargs["control_image"] = pipeline.edge_detector.hed(image)
                            elif upscale_controlnet == "scribble":
                                pipeline.controlnet = "scribble"
                                kwargs["control_image"] = pipeline.edge_detector.hed(image, scribble=True)
                            elif upscale_controlnet == "mlsd":
                                pipeline.controlnet = "mlsd"
                                kwargs["control_image"] = pipeline.line_detector.mlsd(image)
                            elif upscale_controlnet == "line":
                                pipeline.controlnet = "line"
                                kwargs["control_image"] = pipeline.line_detector.line(image)
                            elif upscale_controlnet == "anime":
                                pipeline.controlnet = "anime"
                                kwargs["control_image"] = pipeline.line_detector.line(image, anime=True)
                            elif upscale_controlnet == "tile":
                                pipeline.controlnet = "tile"
                                kwargs["control_image"] = image
                            elif upscale_controlnet == "depth":
                                pipeline.controlnet = "depth"
                                kwargs["control_image"] = pipeline.depth_detector.midas(image)
                            elif upscale_controlnet == "normal":
                                pipeline.controlnet = "normal"
                                kwargs["control_image"] = pipeline.depth_detector.normal(image)
                            else:
                                logger.error(f"Unknown controlnet {upscale_controlnet}, ignoring.")
                                pipeline.controlnet = None
                        else:
                            pipeline.controlnet = None
                        if self.refiner and self.upscale_pipeline != "base":
                            upscale_pipeline = pipeline.refiner_pipeline
                        else:
                            pipeline.reload_pipeline()  # If we didn't change controlnet, then pipeline is still on CPU
                            upscale_pipeline = pipeline.pipeline

                        if self.upscale_diffusion_scale_chunking_size:
                            # Max out at half of the frame size or we get discontinuities
                            kwargs["chunking_size"] = min(
                                self.upscale_diffusion_chunking_size * (j + 1), pipeline.size // 2
                            )
                        if self.upscale_diffusion_scale_chunking_blur:
                            kwargs["chunking_blur"] = min(
                                self.upscale_diffusion_chunking_blur * (j + 1), pipeline.size // 2
                            )

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
                    if self.refiner and self.upscale_pipeline != "base":
                        logger.debug("Offloading refiner for next inference.")
                        pipeline.offload_refiner()
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
        for key in ["image", "control_image", "mask"]:
            if metadata.get(key, None) is not None:
                metadata[key] = self.get_image_metadata(metadata[key])
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
            elif self.outscale > 1:
                upscale_check = self.upscale
                if isinstance(upscale_check, list):
                    upscale_check = upscale_check[0]
                if isinstance(upscale_check, str) and "gan" in upscale_check:
                    next_intention = "upscaling"
                elif self.upscale_diffusion:
                    next_intention = "refining"
            
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
        if image_r_max > 0 and self.prompt:
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
        upscale_dict: Optional[Dict[str, Any]] = None
        if self.upscale:
            upscale_diffusion_dict: Union[bool, Dict[str, Any]] = self.upscale_diffusion
            if self.upscale_diffusion:
                upscale_diffusion_dict = {
                    "steps": self.upscale_diffusion_steps,
                    "guidance_scale": self.upscale_diffusion_guidance_scale,
                    "chunking_size": self.upscale_diffusion_chunking_size,
                    "chunking_blur": self.upscale_diffusion_chunking_blur,
                    "strength": self.upscale_diffusion_strength,
                    "prompt": self.upscale_diffusion_prompt,
                    "prompt_2": self.upscale_diffusion_prompt_2,
                    "negative_prompt": self.upscale_diffusion_negative_prompt,
                    "negative_prompt_2": self.upscale_diffusion_negative_prompt_2,
                    "controlnet": self.upscale_diffusion_controlnet,
                    "scale_chunking_size": self.upscale_diffusion_scale_chunking_size,
                    "scale_chunking_blur": self.upscale_diffusion_scale_chunking_blur,
                }
            upscale_dict = {
                "method": self.upscale,
                "amount": self.outscale,
                "iterative": self.upscale_iterative,
                "diffusion": upscale_diffusion_dict,
                "pipeline": self.upscale_pipeline
            }

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
            "upscale": upscale_dict,
            "chunking_size": self.chunking_size,
            "chunking_blur": self.chunking_blur,
            "build_tensorrt": self.build_tensorrt,
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
            "chunking_blur",
            "samples",
            "iterations",
            "seed",
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "build_tensorrt",
        ]:
            if arg in plan_dict:
                kwargs[arg] = plan_dict[arg]

        if "image" in plan_dict:
            if isinstance(plan_dict["image"], str) and os.path.exists(plan_dict["image"]):
                kwargs["image"] = PIL.Image.open(plan_dict["image"])
            else:
                kwargs["image"] = plan_dict["image"]

        upscale = plan_dict.get("upscale", None)
        if isinstance(upscale, dict):
            kwargs["upscale"] = upscale["method"]
            kwargs["outscale"] = upscale["amount"]
            kwargs["upscale_iterative"] = upscale.get("iterative", False)
            kwargs["upscale_pipeline"] = upscale.get("pipeline", None)
            upscale_diffusion = upscale.get("diffusion", None)
            if isinstance(upscale_diffusion, dict):
                kwargs["upscale_diffusion"] = True
                for arg in [
                    "steps",
                    "controlnet",
                    "guidance_scale",
                    "strength",
                    "chunking_size",
                    "chunking_blur",
                    "prompt",
                    "prompt_2",
                    "negative_prompt",
                    "negative_prompt_2",
                    "scale_chunking_size",
                    "scale_chunking_blur",
                ]:
                    if arg in upscale_diffusion:
                        kwargs[f"upscale_diffusion_{arg}"] = upscale_diffusion[arg]
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
        outscale: Optional[int] = 1,
        upscale: Optional[Union[UPSCALE_LITERAL, List[UPSCALE_LITERAL]]] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_pipeline: Optional[str] = None,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = DEFAULT_UPSCALE_DIFFUSION_STEPS,
        upscale_diffusion_guidance_scale: Optional[
            Union[float, int, List[Union[float, int]]]
        ] = DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE,
        upscale_diffusion_strength: Optional[Union[float, List[float]]] = DEFAULT_UPSCALE_DIFFUSION_STRENGTH,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_PROMPT,
        upscale_diffusion_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_NEGATIVE_PROMPT,
        upscale_diffusion_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_controlnet: Optional[Union[CONTROLNET_LITERAL, List[CONTROLNET_LITERAL]]] = None,
        upscale_diffusion_chunking_size: Optional[int] = None,
        upscale_diffusion_chunking_blur: Optional[int] = None,
        upscale_diffusion_scale_chunking_size: bool = True,
        upscale_diffusion_scale_chunking_blur: bool = True,
        **kwargs: Any,
    ) -> DiffusionPlan:
        """
        Generates a plan to upscale a single image
        """
        if not outscale or outscale < 1 or not upscale:
            raise ValueError("Upscaling requires at least the outscale and upscale method.")
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
                "fit": None,
                "anchor": None,
                "infer": False,
                "inpaint": False,
                "control": False,
                "controlnet": None,
                "prompt": None,
                "prompt_2": None,
                "negative_prompt": None,
                "negative_prompt_2": None,
                "strength": None,
                "conditioning_scale": None,
                "mask": None,
                "process_control_image": None,
                "remove_background": None,
                "invert": None,
                "invert_mask": None,
                "crop_inpaint": None,
                "inpaint_feather": None
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
            upscale=upscale,
            outscale=outscale,
            upscale_iterative=upscale_iterative,
            upscale_diffusion=upscale_diffusion,
            upscale_pipeline=upscale_pipeline,
            upscale_diffusion_steps=upscale_diffusion_steps,
            upscale_diffusion_guidance_scale=upscale_diffusion_guidance_scale,
            upscale_diffusion_strength=upscale_diffusion_strength,
            upscale_diffusion_controlnet=upscale_diffusion_controlnet,
            upscale_diffusion_chunking_size=upscale_diffusion_chunking_size,
            upscale_diffusion_chunking_blur=upscale_diffusion_chunking_blur,
            upscale_diffusion_scale_chunking_size=upscale_diffusion_scale_chunking_size,
            upscale_diffusion_scale_chunking_blur=upscale_diffusion_scale_chunking_blur,
            nodes=nodes,
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
        chunking_blur: Optional[int] = None,
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
        control_image: Optional[Union[str, PIL.Image.Image]] = None,
        controlnet: Optional[CONTROLNET_LITERAL] = None,
        conditioning_scale: Optional[float] = None,
        process_control_image: bool = True,
        remove_background: bool = False,
        fill_background: bool = False,
        scale_to_model_size: bool = True,
        invert_control_image: bool = False,
        invert_mask: bool = False,
        crop_inpaint: bool = True,
        inpaint_feather: int = 32,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        refiner_strength: Optional[float] = DEFAULT_REFINER_STRENGTH,
        refiner_guidance_scale: Optional[float] = DEFAULT_REFINER_GUIDANCE_SCALE,
        refiner_aesthetic_score: Optional[float] = DEFAULT_AESTHETIC_SCORE,
        refiner_negative_aesthetic_score: Optional[float] = DEFAULT_NEGATIVE_AESTHETIC_SCORE,
        refiner_prompt: Optional[str] = None,
        refiner_prompt_2: Optional[str] = None,
        refiner_negative_prompt: Optional[str] = None,
        refiner_negative_prompt_2: Optional[str] = None,
        outscale: Optional[int] = 1,
        upscale: Optional[Union[UPSCALE_LITERAL, List[UPSCALE_LITERAL]]] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_pipeline: Optional[str] = None,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = DEFAULT_UPSCALE_DIFFUSION_STEPS,
        upscale_diffusion_guidance_scale: Optional[Union[float, int, List[Union[float, int]]]] = DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE,
        upscale_diffusion_strength: Optional[Union[float, List[float]]] = DEFAULT_UPSCALE_DIFFUSION_STRENGTH,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_PROMPT,
        upscale_diffusion_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_negative_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_NEGATIVE_PROMPT,
        upscale_diffusion_negative_prompt_2: Optional[Union[str, List[str]]] = None,
        upscale_diffusion_controlnet: Optional[Union[CONTROLNET_LITERAL, List[CONTROLNET_LITERAL]]] = None,
        upscale_diffusion_chunking_size: Optional[int] = None,
        upscale_diffusion_chunking_blur: Optional[int] = None,
        upscale_diffusion_scale_chunking_size: bool = True,
        upscale_diffusion_scale_chunking_blur: bool = True,
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
            upscale=upscale,
            outscale=outscale,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            chunking_size=chunking_size,
            chunking_blur=chunking_blur,
            upscale_iterative=upscale_iterative,
            upscale_diffusion=upscale_diffusion,
            upscale_pipeline=upscale_pipeline,
            upscale_diffusion_steps=upscale_diffusion_steps,
            upscale_diffusion_guidance_scale=upscale_diffusion_guidance_scale,
            upscale_diffusion_strength=upscale_diffusion_strength,
            upscale_diffusion_controlnet=upscale_diffusion_controlnet,
            upscale_diffusion_chunking_size=upscale_diffusion_chunking_size,
            upscale_diffusion_chunking_blur=upscale_diffusion_chunking_blur,
            upscale_diffusion_scale_chunking_size=upscale_diffusion_scale_chunking_size,
            upscale_diffusion_scale_chunking_blur=upscale_diffusion_scale_chunking_blur,
            nodes=[],
        )

        # We'll assemble multiple token sets for overall diffusion
        upscale_diffusion_prompt_tokens = [TokenMerger()]
        upscale_diffusion_prompt_2_tokens = [TokenMerger()]
        upscale_diffusion_negative_prompt_tokens = [TokenMerger()]
        upscale_diffusion_negative_prompt_2_tokens = [TokenMerger()]
        refiner_prompt_tokens = TokenMerger()
        refiner_prompt_2_tokens = TokenMerger()
        refiner_negative_prompt_tokens = TokenMerger()
        refiner_negative_prompt_2_tokens = TokenMerger()

        if upscale_diffusion_prompt:
            if isinstance(upscale_diffusion_prompt, list):
                upscale_diffusion_prompt = [
                    upscale_prompt for upscale_prompt in upscale_diffusion_prompt if upscale_prompt
                ]
                for i, upscale_prompt in enumerate(upscale_diffusion_prompt):
                    if len(upscale_diffusion_prompt_tokens) < i + 1:
                        upscale_diffusion_prompt_tokens.append(TokenMerger())
                    upscale_diffusion_prompt_tokens[i].add(upscale_prompt)
            else:
                upscale_diffusion_prompt_tokens[0].add(upscale_diffusion_prompt)
        if upscale_diffusion_prompt_2:
            if isinstance(upscale_diffusion_prompt_2, list):
                upscale_diffusion_prompt_2 = [
                    upscale_prompt for upscale_prompt in upscale_diffusion_prompt_2 if upscale_prompt
                ]
                for i, upscale_prompt in enumerate(upscale_diffusion_prompt_2):
                    if len(upscale_diffusion_prompt_2_tokens) < i + 1:
                        upscale_diffusion_prompt_2_tokens.append(TokenMerger())
                    upscale_diffusion_prompt_2_tokens[i].add(upscale_prompt)
            else:
                upscale_diffusion_prompt_2_tokens[0].add(upscale_diffusion_prompt_2)
        
        if prompt:
            for token_merger in upscale_diffusion_prompt_tokens:
                token_merger.add(prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if prompt_2:
            for token_merger in upscale_diffusion_prompt_tokens:
                token_merger.add(prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT)

        if model_prompt:
            refiner_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)
            for token_merger in upscale_diffusion_prompt_tokens:
                token_merger.add(model_prompt, MODEL_PROMPT_WEIGHT)
        if model_prompt_2:
            refiner_prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)
            for token_merger in upscale_diffusion_prompt_2_tokens:
                token_merger.add(model_prompt_2, MODEL_PROMPT_WEIGHT)

        if upscale_diffusion_negative_prompt:
            if isinstance(upscale_diffusion_negative_prompt, list):
                upscale_diffusion_negative_prompt = [
                    upscale_negative_prompt
                    for upscale_negative_prompt in upscale_diffusion_negative_prompt
                    if upscale_negative_prompt
                ]
                for i, negative_prompt in enumerate(upscale_diffusion_negative_prompt):
                    if len(upscale_diffusion_negative_prompt_tokens) < i + 1:
                        upscale_diffusion_negative_prompt_tokens.append(TokenMerger())
                    upscale_diffusion_negative_prompt_tokens[i].add(negative_prompt)
            else:
                upscale_diffusion_negative_prompt_tokens[0].add(upscale_diffusion_negative_prompt)
        if upscale_diffusion_negative_prompt_2:
            if isinstance(upscale_diffusion_negative_prompt_2, list):
                upscale_diffusion_negative_prompt_2 = [
                    upscale_negative_prompt
                    for upscale_negative_prompt in upscale_diffusion_negative_prompt_2
                    if upscale_negative_prompt
                ]
                for i, negative_prompt in enumerate(upscale_diffusion_negative_prompt_2):
                    if len(upscale_diffusion_negative_prompt_2_tokens) < i + 1:
                        upscale_diffusion_negative_prompt_2_tokens.append(TokenMerger())
                    upscale_diffusion_negative_prompt_2_tokens[i].add(negative_prompt)
            else:
                upscale_diffusion_negative_prompt_2_tokens[0].add(upscale_diffusion_negative_prompt_2)

        if negative_prompt:
            for token_merger in upscale_diffusion_negative_prompt_tokens:
                token_merger.add(negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if negative_prompt_2:
            for token_merger in upscale_diffusion_negative_prompt_2_tokens:
                token_merger.add(negative_prompt_2, GLOBAL_PROMPT_UPSCALE_WEIGHT)

        if model_negative_prompt:
            refiner_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
            for token_merger in upscale_diffusion_negative_prompt_tokens:
                token_merger.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
        if model_negative_prompt_2:
            refiner_negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)
            for token_merger in upscale_diffusion_negative_prompt_2_tokens:
                token_merger.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)

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
                name = "Image to Image"
                if isinstance(image, str):
                    image = PIL.Image.open(image)
                if width and height:
                    image = fit_image(image, width, height, fit, anchor)
                else:
                    width, height = image.size
                
            if mask:
                name = "Text to Image"
                if isinstance(mask, str):
                    mask = PIL.Image.open(mask)
                if width and height:
                    mask = fit_image(mask, width, height, fit, anchor)
                else:
                    width, height = mask.size
            
            if control_image:
                name = "Controlled Image to Image" if image else "Controlled Text to Image"
                if isinstance(control_image, str):
                    control_image = PIL.Image.open(control_image)
                if width and height:
                    control_image = fit_image(control_image, width, height, fit, anchor)
                else:
                    width, height = control_image.size

            if mask and invert_mask:
                mask = PIL.ImageOps.invert(mask.convert("L"))
            
            if control_image and not process_control_image and invert_control_image:
                control_image = PIL.ImageOps.invert(control_image)
            
            if image and remove_background and fill_background:
                image = execute_remove_background(image)
                remove_background = False
                
                if not mask:
                    mask = PIL.Image.new("RGB", image.size, (255, 255, 255))

                alpha = image.split()[-1]
                black = PIL.Image.new("RGB", image.size, (0, 0, 0))
                mask.paste(black, mask=alpha)

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
                process_control_image=process_control_image,
                scale_to_model_size=scale_to_model_size,
                controlnet=controlnet,
                control_image=control_image,
                conditioning_scale=conditioning_scale,
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
            plan.upscale_diffusion_prompt = [
                str(merger) for merger in upscale_diffusion_prompt_tokens
            ]
            plan.upscale_diffusion_prompt_2 = [
                str(merger) for merger in upscale_diffusion_prompt_2_tokens
            ]
            plan.upscale_diffusion_negative_prompt = [
                str(merger) for merger in upscale_diffusion_negative_prompt_tokens
            ]
            plan.upscale_diffusion_negative_prompt_2 = [
                str(merger) for merger in upscale_diffusion_negative_prompt_2_tokens
            ]
            return plan

        # Using the diffusion canvas, assemble a multi-step plan
        for i, node_dict in enumerate(nodes):
            step = DiffusionStep(
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
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

            node_infer = node_dict.get("infer", False)
            node_inpaint = node_dict.get("inpaint", False)
            node_control = node_dict.get("control", False)

            node_controlnet = node_dict.get("controlnet", None)
            node_prompt = node_dict.get("prompt", None)
            node_prompt_2 = node_dict.get("prompt_2", None)
            node_negative_prompt = node_dict.get("negative_prompt", None)
            node_negative_prompt_2 = node_dict.get("negative_prompt_2", None)
            node_strength: Optional[float] = node_dict.get("strength", None)
            node_conditioning_scale: Optional[float] = node_dict.get("conditioning_scale", None)
            node_image = node_dict.get("image", None)
            node_inpaint_mask = node_dict.get("mask", None)
            node_crop_inpaint = node_dict.get("crop_inpaint", crop_inpaint)
            node_inpaint_feather = node_dict.get("inpaint_feather", inpaint_feather)
            node_invert = node_dict.get("invert", False)
            node_invert_mask = node_dict.get("invert_mask", False)
            node_process_control_image = node_dict.get("process_control_image", True)
            node_scale_to_model_size = node_dict.get("scale_to_model_size", True)
            node_remove_background = bool(node_dict.get("remove_background", False))
            node_inference_steps: Optional[int] = node_dict.get("inference_steps", None)  # type: ignore[assignment]
            node_guidance_scale: Optional[float] = node_dict.get("guidance_scale", None)  # type: ignore[assignment]

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
            else:
                raise ValueError(f"Node {i} missing width, pass 'w' or an image")
            if "h" in node_dict:
                node_height = int(node_dict["h"])
            elif node_image is not None:  # type: ignore[unreachable]
                _, node_height = node_image.size
            else:
                raise ValueError(f"Node {i} missing height, pass 'h' or an image")

            node_bounds = [
                (node_left, node_top),
                (node_left + node_width, node_top + node_height),
            ]

            if node_prompt:
                node_prompt_tokens.add(node_prompt)
                for merger in upscale_diffusion_prompt_tokens:
                    merger.add(node_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if prompt and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_prompt_tokens.add(prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_prompt:
                node_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)

            if node_prompt_2:
                node_prompt_2_tokens.add(node_prompt_2)
                for merger in upscale_diffusion_prompt_2_tokens:
                    merger.add(node_prompt_2, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if prompt_2 and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_prompt_2_tokens.add(prompt_2, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_prompt_2:
                node_prompt_2_tokens.add(model_prompt_2, MODEL_PROMPT_WEIGHT)

            if node_negative_prompt:
                node_negative_prompt_tokens.add(node_negative_prompt)
                for merger in upscale_diffusion_negative_prompt_tokens:
                    merger.add(node_negative_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if negative_prompt and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_negative_prompt_tokens.add(negative_prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_negative_prompt:
                node_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)
            
            if node_negative_prompt_2:
                node_negative_prompt_tokens.add(node_negative_prompt_2)
                for merger in upscale_diffusion_negative_prompt_2_tokens:
                    merger.add(node_negative_prompt_2, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if negative_prompt_2 and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_negative_prompt_2_tokens.add(negative_prompt_2, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_negative_prompt_2:
                node_negative_prompt_2_tokens.add(model_negative_prompt_2, MODEL_PROMPT_WEIGHT)

            if node_image:
                if node_remove_background:
                    node_image = execute_remove_background(node_image)

                # Resize node image to nearest multiple of 8
                node_image_width, node_image_height = node_image.size
                node_image_width = 8 * math.ceil(node_image_width / 8)
                node_image_height = 8 * math.ceil(node_image_height / 8)
                node_image = node_image.resize((node_image_width, node_image_height))

                node_image = fit_image(node_image, node_width, node_height, node_fit, node_anchor)
                node_mask = PIL.Image.new("RGB", node_image.size, (255, 255, 255))
                node_alpha = node_image.split()[-1]
                node_alpha_clamp = PIL.Image.eval(node_alpha, lambda a: 255 if a > 128 else 0)

                black = PIL.Image.new("RGB", node_image.size, (0, 0, 0))
                white = PIL.Image.new("RGB", node_image.size, (255, 255, 255))
                node_mask.paste(black, mask=node_alpha)
                node_mask_r_min, node_mask_r_max = node_mask.getextrema()[1]

                node_image_needs_outpainting = node_mask_r_max > 0

                if node_inpaint:
                    if node_inpaint_mask:
                        # Inpaint prior to anything else.
                        node_inpaint_mask = node_inpaint_mask.convert("L")
                        if node_invert_mask:
                            node_inpaint_mask = PIL.ImageOps.invert(node_inpaint_mask)
                    else:
                        # Make blank mask
                        node_inpaint_mask = PIL.Image.new("L", node_image.size, 0)
                    if node_image_needs_outpainting:
                        # Merge inpaint and outpaint masks
                        node_mask.paste(node_inpaint_mask)
                        node_inverse_alpha = PIL.Image.eval(node_alpha_clamp, lambda a: 255 - a)
                        node_mask.paste(white, mask=node_inverse_alpha)
                    else:
                        # Just use inpaint mask
                        node_mask.paste(node_inpaint_mask)

                    inpaint_image_step = DiffusionStep(
                        name=f"Inpaint Node {i+1}",
                        image=node_image,
                        mask=feather_mask(node_mask.convert("1")),
                        prompt=str(node_prompt_tokens),
                        prompt_2=str(node_prompt_2_tokens),
                        negative_prompt=str(node_negative_prompt_tokens),
                        negative_prompt_2=str(node_negative_prompt_2_tokens),
                        guidance_scale=guidance_scale,
                        num_inference_steps=node_inference_steps if node_inference_steps else num_inference_steps,
                        crop_inpaint=node_crop_inpaint,
                        inpaint_feather=node_inpaint_feather,
                        refiner_strength=refiner_strength,
                        refiner_guidance_scale=refiner_guidance_scale,
                        refiner_aesthetic_score=refiner_aesthetic_score,
                        refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                        refiner_prompt=refiner_prompt,
                        refiner_prompt_2=refiner_prompt_2,
                        refiner_negative_prompt=refiner_negative_prompt,
                        refiner_negative_prompt_2=refiner_negative_prompt_2,
                    )
                    step_image = inpaint_image_step
                elif node_image_needs_outpainting and (node_infer or node_control):
                    # There are gaps; add an outpaint step before the infer/control
                    if node_infer or node_process_control_image:
                        outpaint_image_step = DiffusionStep(
                            name=f"In-place Outpaint Node {i+1}",
                            image=node_image,
                            mask=feather_mask(node_mask.convert("1")),
                            prompt=str(node_prompt_tokens),
                            prompt_2=str(node_prompt_2_tokens),
                            negative_prompt=str(node_negative_prompt_tokens),
                            negative_prompt_2=str(node_negative_prompt_2_tokens),
                            crop_inpaint=node_crop_inpaint,
                            inpaint_feather=node_inpaint_feather,
                            guidance_scale=guidance_scale,
                            num_inference_steps=node_inference_steps if node_inference_steps else num_inference_steps,
                            refiner_strength=refiner_strength,
                            refiner_guidance_scale=refiner_guidance_scale,
                            refiner_aesthetic_score=refiner_aesthetic_score,
                            refiner_negative_aesthetic_score=refiner_negative_aesthetic_score,
                            refiner_prompt=refiner_prompt,
                            refiner_prompt_2=refiner_prompt_2,
                            refiner_negative_prompt=refiner_negative_prompt,
                            refiner_negative_prompt_2=refiner_negative_prompt_2,
                        )
                        step_image = outpaint_image_step
                    elif node_invert:
                        # Paste on white
                        white = PIL.Image.new("RGB", node_image.size, (255, 255, 255))
                        white.paste(node_image, mask=node_alpha)
                        step_image = white
                    else:
                        # Paste on black
                        black = PIL.Image.new("RGB", node_image.size, (0, 0, 0))
                        black.paste(node_image)
                        step_image = black
                else:
                    step_image = node_image

                if not node_infer and not node_inpaint and not node_control:
                    # Just paste image
                    step.name = f"Pass-through Node {i+1}"
                    step.image = step_image
                elif not node_infer and node_inpaint and not node_control:
                    # Just inpaint image
                    step = step_image
                    step.name = f"Inpaint Node {i+1}"
                else:
                    # Some combination of ops
                    step.prompt = str(node_prompt_tokens)
                    step.prompt_2 = str(node_prompt_2_tokens)
                    step.negative_prompt = str(node_negative_prompt_tokens)
                    step.negative_prompt_2 = str(node_negative_prompt_2_tokens)
                    if node_strength:
                        step.strength = node_strength
                    if node_infer:
                        step.image = step_image
                        step.name = f"Image to Image Node {i+1}"
                    if node_control and node_controlnet:
                        if node_infer:
                            step.name = f"Controlled Image to Image Node {i+1}"
                        else:
                            step.name = f"Controlled Text to Image Node {i+1}"
                        step.controlnet = node_controlnet  # type: ignore
                        step.control_image = step_image
                        if not node_process_control_image:
                            step.process_control_image = False
                            if node_invert and not isinstance(step_image, DiffusionStep):
                                step.control_image = PIL.ImageOps.invert(step_image.convert("L"))  # type: ignore[unreachable]
                        if node_conditioning_scale:
                            step.conditioning_scale = node_conditioning_scale
            elif node_prompt:
                step.name = f"Region Prompt Node {i+1}"
                step.prompt = str(node_prompt_tokens)
                step.prompt_2 = str(node_prompt_2_tokens)
                step.negative_prompt = str(node_negative_prompt_tokens)
                step.negative_prompt_2 = str(node_negative_prompt_2_tokens)
                step.width = node_width
                step.height = node_height
                step.remove_background = node_remove_background
                step.refiner_prompt = refiner_prompt
                step.refiner_prompt_2 = refiner_prompt_2
                step.refiner_negative_prompt = refiner_negative_prompt
                step.refiner_negative_prompt_2 = refiner_negative_prompt_2
            else:
                raise ValueError("Can't assemble a node from arguments")

            # Set common args for node
            if node_inference_steps:
                step.num_inference_steps = node_inference_steps
            if node_guidance_scale:
                step.guidance_scale = node_guidance_scale
            if node_refiner_strength:
                step.refiner_strength = node_refiner_strength
            if node_refiner_guidance_scale:
                step.refiner_guidance_scale = node_refiner_guidance_scale
            if node_refiner_aesthetic_score:
                step.refiner_aesthetic_score = node_refiner_aesthetic_score
            if node_refiner_negative_aesthetic_score:
                step.refiner_negative_aesthetic_score = node_refiner_negative_aesthetic_score
            if not node_scale_to_model_size:
                step.scale_to_model_size = False
            # Add step to plan
            plan.nodes.append(DiffusionNode(node_bounds, step))

        plan.upscale_diffusion_prompt = [
            str(prompt) for prompt in upscale_diffusion_prompt_tokens
        ]
        plan.upscale_diffusion_prompt_2 = [
            str(prompt) for prompt in upscale_diffusion_prompt_2_tokens
        ]
        plan.upscale_diffusion_negative_prompt = [
            str(negative_prompt) for negative_prompt in upscale_diffusion_negative_prompt_tokens
        ]
        plan.upscale_diffusion_negative_prompt_2 = [
            str(negative_prompt) for negative_prompt in upscale_diffusion_negative_prompt_2_tokens
        ]

        return plan
