from __future__ import annotations

import math
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps

from typing import (
    Optional,
    Dict,
    Any,
    Union,
    Tuple,
    List,
    Literal,
    Callable,
    TypedDict,
    TYPE_CHECKING,
)

from enfugue.util import logger, feather_mask, fit_image, remove_background, TokenMerger

if TYPE_CHECKING:
    from enfugue.diffusers.manager import DiffusionPipelineManager
    from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

DEFAULT_SIZE = 512
DEFAULT_IMAGE_CALLBACK_STEPS = 10
DEFAULT_CONDITIONING_SCALE = 1.0
DEFAULT_CHUNKING_SIZE = 64
DEFAULT_CHUNKING_BLUR = 64
DEFAULT_IMG2IMG_STRENGTH = 0.8
DEFAULT_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_UPSCALE_PROMPT = (
    "highly detailed, ultra-detailed, intricate detail, high definition, HD, 4k, ultra high def"
)
DEFAULT_UPSCALE_NEGATIVE_PROMPT = "blurry, out-of-focus, ugly, pixelated, artifacts"
DEFAULT_UPSCALE_DIFFUSION_STEPS = 100
DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE = 12
DEFAULT_UPSCALE_DIFFUSION_STRENGTH = 0.2
DEFAULT_UPSCALE_DIFFUSION_CHUNKING_SIZE = 64
DEFAULT_UPSCALE_DIFFUSION_CHUNKING_BLUR = 64

MODEL_PROMPT_WEIGHT = 0.2
GLOBAL_PROMPT_STEP_WEIGHT = 0.4
GLOBAL_PROMPT_UPSCALE_WEIGHT = 0.4
UPSCALE_PROMPT_STEP_WEIGHT = 0.1
MAX_IMAGE_SCALE = 3.0

__all__ = ["DiffusionStep", "DiffusionPlan"]


class NodeDict(TypedDict):
    w: int
    h: int
    x: int
    y: int
    fit: Optional[Literal["actual", "stretch", "contain", "cover"]]
    anchor: Optional[
        Literal[
            "top-left",
            "top-center",
            "top-right",
            "center-left",
            "center-center",
            "center-right",
            "bottom-left",
            "bottom-center",
            "bottom-right",
        ]
    ]
    infer: Optional[bool]
    inpaint: Optional[bool]
    control: Optional[bool]
    controlnet: Optional[str]
    prompt: Optional[str]
    negative_prompt: Optional[str]
    strength: Optional[float]
    conditioning_scale: Optional[float]
    image: Optional[PIL.Image.Image]
    mask: Optional[PIL.Image.Image]


class DiffusionStep:
    """
    A step represents most of the inputs to describe what the image is and how to control inference
    """

    result: StableDiffusionPipelineOutput

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        mask: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        control_image: Optional[Union[DiffusionStep, PIL.Image.Image, str]] = None,
        controlnet: Optional[Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"]] = None,
        conditioning_scale: Optional[float] = DEFAULT_CONDITIONING_SCALE,
        strength: Optional[float] = DEFAULT_IMG2IMG_STRENGTH,
        num_inference_steps: Optional[int] = DEFAULT_INFERENCE_STEPS,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        remove_background: bool = False,
        process_control_image: bool = True,
        scale_to_model_size: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.image = image
        self.mask = mask
        self.control_image = control_image
        self.controlnet = controlnet
        self.strength = strength if strength is not None else DEFAULT_IMG2IMG_STRENGTH
        self.conditioning_scale = (
            conditioning_scale if conditioning_scale is not None else DEFAULT_CONDITIONING_SCALE
        )
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else DEFAULT_INFERENCE_STEPS
        )
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE
        )
        self.remove_background = remove_background
        self.process_control_image = process_control_image
        self.scale_to_model_size = scale_to_model_size

    def get_serialization_dict(self) -> Dict[str, Any]:
        """
        Gets the dictionary that will be returned to serialize
        """
        serialized: Dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "controlnet": self.controlnet,
            "conditioning_scale": self.conditioning_scale,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "remove_background": self.remove_background,
            "process_control_image": self.process_control_image,
            "scale_to_model_size": self.scale_to_model_size,
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
            else:
                serialized[key] = child

        serialized["children"] = [child.get_serialization_dict() for child in serialize_children]
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
            "negative_prompt": self.negative_prompt,
            "image": self.image,
            "control_image": self.control_image,
            "conditioning_scale": self.conditioning_scale,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }

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
            if self.controlnet == "mlsd":
                return pipeline.edge_detector.mlsd(control_image)
            if self.controlnet == "hed":
                return pipeline.edge_detector.hed(control_image)
        return control_image

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        **kwargs: Any,
    ) -> StableDiffusionPipelineOutput:
        """
        Executes this pipeline step.
        """
        if hasattr(self, "result"):
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
                samples = kwargs.get("samples", 1)
                from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

                self.result = StableDiffusionPipelineOutput(
                    images=[image] * samples, nsfw_content_detected=[False] * samples
                )
                return self.result
            raise ValueError("No prompt or image in this step; cannot invoke or pass through.")

        pipeline.controlnet = self.controlnet
        invocation_kwargs = {**kwargs, **self.kwargs}

        image_scale = 1
        image_width, image_height = None, None

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
            invocation_kwargs["mask"] = mask
            if image is not None:
                assert image.size == mask.size
            if control_image is not None:
                assert control_image.size == mask.size
            else:
                image_width, image_height = mask.size
        if (
            self.width is not None
            and self.height is not None
            and image_width is None
            and image_height is None
        ):
            image_width, image_height = self.width, self.height

        if image_width is None or image_height is None:
            raise ValueError("No known invocation size.")

        if image_width is not None and image_width < pipeline.size:
            image_scale = pipeline.size / image_width
        if image_height is not None and image_height < pipeline.size:
            image_scale = max(image_scale, pipeline.size / image_height)

        if image_scale > MAX_IMAGE_SCALE or not self.scale_to_model_size:
            # Refuse it's too oblong. We'll just calculate at the appropriate size.
            image_scale = 1

        invocation_kwargs["width"] = 8 * round((image_width * image_scale) / 8)
        invocation_kwargs["height"] = 8 * round((image_height * image_scale) / 8)
        invocation_kwargs["control_image"] = self.check_process_control_image(
            pipeline, control_image
        )

        if image_scale > 1:
            # scale input images up
            for key in ["image", "mask", "control_image"]:
                if invocation_kwargs.get(key, None) is not None:
                    invocation_kwargs[key] = self.scale_image(invocation_kwargs[key], image_scale)

        result = pipeline(**invocation_kwargs)

        if self.remove_background:
            for i, image in enumerate(result["images"]):
                result["images"][i] = remove_background(image)
                logger.critical(result["images"][i])

        if image_scale > 1:
            for i, image in enumerate(result["images"]):
                result["images"][i] = self.scale_image(image, 1 / image_scale)
                logger.critical(result["images"][i])

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
            "prompt",
            "negative_prompt",
            "controlnet",
            "conditioning_scale",
            "strength",
            "num_inference_steps",
            "guidance_scale",
            "width",
            "height",
            "remove_background",
            "process_control_image",
            "scale_to_model_size",
        ]:
            if key in step_dict:
                kwargs[key] = step_dict[key]

        deserialized_children = [
            DiffusionStep.deserialize_dict(child) for child in step_dict.get("children", [])
        ]
        for key in ["image", "control_image", "mask"]:
            if key not in step_dict:
                continue
            if isinstance(step_dict[key], int):
                kwargs[key] = deserialized_children[step_dict[key]]
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

    def get_serialization_dict(self) -> Dict[str, Any]:
        """
        Gets the step's dict and adds bounds.
        """
        step_dict = self.step.get_serialization_dict()
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
        negative_prompt: Optional[str] = None,  # Global
        size: Optional[int] = DEFAULT_SIZE,
        model: Optional[str] = None,
        lora: Optional[
            Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]
        ] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        width: Optional[int] = DEFAULT_SIZE,
        height: Optional[int] = DEFAULT_SIZE,
        image_callback_steps: Optional[int] = DEFAULT_IMAGE_CALLBACK_STEPS,
        nodes: List[DiffusionNode] = [],
        image: Optional[Union[str, PIL.Image.Image]] = None,
        chunking_size: Optional[int] = DEFAULT_CHUNKING_SIZE,
        chunking_blur: Optional[int] = DEFAULT_CHUNKING_BLUR,
        samples: Optional[int] = 1,
        seed: Optional[int] = None,
        build_tensorrt: bool = False,
        outscale: Optional[int] = 1,
        upscale: Optional[
            Union[
                Literal[
                    "esrgan",
                    "esrganime",
                    "gfpgan",
                    "lanczos",
                    "bilinear",
                    "bicubic",
                    "lanczos",
                    "nearest",
                ],
                List[
                    Literal[
                        "esrgan",
                        "esrganime",
                        "gfpgan",
                        "lanczos",
                        "bilinear",
                        "bicubic",
                        "lanczos",
                        "nearest",
                    ]
                ],
            ]
        ] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = DEFAULT_UPSCALE_DIFFUSION_STEPS,
        upscale_diffusion_guidance_scale: Optional[
            Union[float, int, List[Union[float, int]]]
        ] = DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE,
        upscale_diffusion_strength: Optional[
            Union[float, List[float]]
        ] = DEFAULT_UPSCALE_DIFFUSION_STRENGTH,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_PROMPT,
        upscale_diffusion_negative_prompt: Optional[
            Union[str, List[str]]
        ] = DEFAULT_UPSCALE_NEGATIVE_PROMPT,
        upscale_diffusion_controlnet: Optional[
            Union[
                Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"],
                List[Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"]],
            ]
        ] = None,
        upscale_diffusion_chunking_size: Optional[int] = DEFAULT_UPSCALE_DIFFUSION_CHUNKING_SIZE,
        upscale_diffusion_chunking_blur: Optional[int] = DEFAULT_UPSCALE_DIFFUSION_CHUNKING_BLUR,
        upscale_diffusion_scale_chunking_size: bool = True,
        upscale_diffusion_scale_chunking_blur: bool = True,
    ) -> None:
        self.size = size if size is not None else DEFAULT_SIZE
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.model = model
        self.lora = lora
        self.inversion = inversion
        self.width = width if width is not None else DEFAULT_SIZE
        self.height = height if height is not None else DEFAULT_SIZE
        self.image = image
        self.image_callback_steps = image_callback_steps
        self.chunking_size = (
            chunking_size if chunking_size is not None else DEFAULT_CHUNKING_SIZE
        )  # Pass 0 to disable
        self.chunking_blur = (
            chunking_blur if chunking_blur is not None else DEFAULT_CHUNKING_BLUR
        )  # Pass 0 to disable
        self.samples = samples if samples is not None else 1
        self.seed = seed
        self.build_tensorrt = build_tensorrt
        self.nodes = nodes
        self.outscale = outscale if outscale is not None else 1
        self.upscale = upscale
        self.upscale_iterative = bool(upscale_iterative)
        self.upscale_diffusion = bool(upscale_diffusion)
        self.upscale_diffusion_chunking_size = (
            upscale_diffusion_chunking_size
            if upscale_diffusion_chunking_size is not None
            else DEFAULT_UPSCALE_DIFFUSION_CHUNKING_SIZE
        )
        self.upscale_diffusion_chunking_blur = (
            upscale_diffusion_chunking_blur
            if upscale_diffusion_chunking_blur is not None
            else DEFAULT_UPSCALE_DIFFUSION_CHUNKING_BLUR
        )
        self.upscale_diffusion_guidance_scale = (
            upscale_diffusion_guidance_scale
            if upscale_diffusion_guidance_scale is not None
            else DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE
        )
        self.upscale_diffusion_steps = (
            upscale_diffusion_steps
            if upscale_diffusion_steps is not None
            else DEFAULT_UPSCALE_DIFFUSION_STEPS
        )
        self.upscale_diffusion_strength = (
            upscale_diffusion_strength
            if upscale_diffusion_strength is not None
            else DEFAULT_UPSCALE_DIFFUSION_STRENGTH
        )
        self.upscale_diffusion_prompt = (
            upscale_diffusion_prompt
            if upscale_diffusion_prompt is not None
            else DEFAULT_UPSCALE_PROMPT
        )
        self.upscale_diffusion_negative_prompt = (
            upscale_diffusion_negative_prompt
            if upscale_diffusion_negative_prompt is not None
            else DEFAULT_UPSCALE_NEGATIVE_PROMPT
        )
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

        images, nsfw = self.execute_nodes(
            pipeline, progress_callback, image_callback, image_callback_steps
        )
        if self.upscale is not None and self.outscale > 1:
            if self.upscale_iterative:
                scales = [2] * int(math.log(self.outscale, 2))
            else:
                scales = [self.outscale]
            for j, scale in enumerate(scales):
                # Unload the pipeline to force buffers to refresh, saves VRAM but takes time
                pipeline.unload_pipeline()

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

                    upscale = get_item_for_scale(self.upscale)
                    logger.debug(f"Upscaling sample {i} by {scale} using {upscale}")

                    if upscale == "esrgan":
                        image = pipeline.upscaler.esrgan(image, tile=pipeline.size, outscale=scale)
                    elif upscale == "esrganime":
                        image = pipeline.upscaler.esrgan(
                            image, tile=pipeline.size, outscale=scale, anime=True
                        )
                    elif upscale == "gfpgan":
                        image = pipeline.upscaler.gfpgan(image, tile=pipeline.size, outscale=scale)
                    elif upscale in PIL_INTERPOLATION:
                        width, height = image.size
                        image = image.resize(
                            (width * scale, height * scale), resample=PIL_INTERPOLATION[upscale]
                        )
                    else:
                        logger.error(f"Unknown upscaler {upscale}")
                        return images
                    images[i] = image

                # Unload the upscaler to save VRAM
                pipeline.unload_upscaler()

                if image_callback is not None and (j < len(scales) - 1 or self.upscale_diffusion):
                    image_callback(images)

                if self.upscale_diffusion:
                    # Disable pipeline safety here, it gives many false positives when upscaling.
                    # We'll re-enable it after.
                    re_enable_safety = pipeline.safe
                    pipeline.safe = False
                    for i, image in enumerate(images):
                        if nsfw is not None and nsfw[i]:
                            logger.debug(f"Image {i} had NSFW content, not upscaling.")
                            continue
                        # Prepare the pipeline again
                        self.prepare_pipeline(pipeline)
                        width, height = image.size
                        kwargs = {
                            "width": width,
                            "height": height,
                            "image": image,
                            "num_images_per_prompt": 1,
                            "prompt": get_item_for_scale(self.upscale_diffusion_prompt),
                            "negative_prompt": get_item_for_scale(
                                self.upscale_diffusion_negative_prompt
                            ),
                            "strength": get_item_for_scale(self.upscale_diffusion_strength),
                            "num_inference_steps": get_item_for_scale(self.upscale_diffusion_steps),
                            "guidance_scale": get_item_for_scale(
                                self.upscale_diffusion_guidance_scale
                            ),
                            "chunking_size": self.upscale_diffusion_chunking_size,
                            "chunking_blur": self.upscale_diffusion_chunking_blur,
                            "progress_callback": progress_callback,
                        }
                        if self.upscale_diffusion_scale_chunking_size:
                            # Max out at half of the frame size or we get discontinuities
                            kwargs["chunking_size"] = min(
                                self.upscale_diffusion_chunking_size * (j + 1), pipeline.size // 2
                            )
                        if self.upscale_diffusion_scale_chunking_blur:
                            kwargs["chunking_blur"] = min(
                                self.upscale_diffusion_chunking_blur * (j + 1), pipeline.size // 2
                            )
                        upscale_controlnet = get_item_for_scale(self.upscale_diffusion_controlnet)
                        if upscale_controlnet is not None:
                            logger.debug(f"Enabling {upscale_controlnet} for upscale diffusion")
                            if upscale_controlnet == "canny":
                                pipeline.controlnet = "canny"
                                kwargs["control_image"] = pipeline.edge_detector.canny(image)
                            elif upscale_controlnet == "mlsd":
                                pipeline.controlnet = "mlsd"
                                kwargs["control_image"] = pipeline.edge_detector.mlsd(image)
                            elif upscale_controlnet == "hed":
                                pipeline.controlnet = "hed"
                                kwargs["control_image"] = pipeline.edge_detector.hed(image)
                            elif upscale_controlnet == "tile":
                                pipeline.controlnet = "tile"
                                kwargs["control_image"] = image
                            else:
                                logger.error(f"Unknown controlnet {upscale_controlnet}, ignoring.")
                                pipeline.controlnet = None
                        else:
                            pipeline.controlnet = None
                        image = pipeline(**kwargs).images[0]
                        images[i] = image
                    if re_enable_safety:
                        pipeline.safe = True
                if j < len(scales) - 1 and image_callback is not None:
                    image_callback(images)
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=nsfw)

    def prepare_pipeline(self, pipeline: DiffusionPipelineManager) -> None:
        """
        Assigns pipeline-level variables.
        """
        pipeline.model = self.model
        pipeline.lora = self.lora
        pipeline.inversion = self.inversion
        pipeline.size = self.size
        if self.build_tensorrt:
            pipeline.build_tensorrt = True

    def execute_nodes(
        self,
        pipeline: DiffusionPipelineManager,
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

        # Define callback kwargs
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

        images = [PIL.Image.new("RGBA", (self.width, self.height)) for i in range(self.samples)]
        image_draw = [PIL.ImageDraw.Draw(image) for image in images]
        nsfw_content_detected = [False] * self.samples

        # Keep a final mask of all nodes to outpaint in the end
        outpaint_mask = PIL.Image.new("RGB", (self.width, self.height), (255, 255, 255))
        outpaint_draw = PIL.ImageDraw.Draw(outpaint_mask)

        for i, node in enumerate(self.nodes):
            if image_callback is not None:

                def node_image_callback(callback_images: List[PIL.Image.Image]) -> None:
                    for j, callback_image in enumerate(callback_images):
                        images[j].paste(node.resize_image(callback_image), node.bounds[0])
                    image_callback(images)  # type: ignore

            else:
                node_image_callback = None  # type: ignore
            invocation_kwargs = {**self.kwargs, **callback_kwargs}
            result = node.execute(
                pipeline, latent_callback=node_image_callback, **invocation_kwargs
            )
            for j, image in enumerate(result["images"]):
                image = node.resize_image(image)
                if image.mode == "RGBA":
                    # Draw the alpha mask of the return image onto the outpaint mask
                    alpha = image.split()[-1]
                    black = PIL.Image.new("RGB", alpha.size, (0, 0, 0))
                    outpaint_mask.paste(black, node.bounds[0], mask=alpha)
                    image_draw[j].rectangle((*node.bounds[0], *node.bounds[1]), fill=(0, 0, 0, 0))
                    images[j].paste(image, node.bounds[0], mask=alpha)
                else:
                    # Draw a rectangle directly
                    outpaint_draw.rectangle(node.bounds, fill="#000000")
                    images[j].paste(node.resize_image(image), node.bounds[0])

                nsfw_content_detected[j] = nsfw_content_detected[j] or (
                    "nsfw_content_detected" in result and result["nsfw_content_detected"][j]
                )

            # Call the callback
            if image_callback is not None:
                image_callback(images)

        # Determine if there's anything left to outpaint
        image_r_min, image_r_max = outpaint_mask.getextrema()[1]
        if image_r_max > 0:
            # Outpaint
            del invocation_kwargs["num_images_per_prompt"]
            outpaint_mask = feather_mask(outpaint_mask)

            outpaint_prompt_tokens = TokenMerger()
            outpaint_negative_prompt_tokens = TokenMerger()

            for i, node in enumerate(self.nodes):
                if node.step.prompt is not None:
                    outpaint_prompt_tokens.add(node.step.prompt)
                if node.step.negative_prompt is not None:
                    outpaint_negative_prompt_tokens.add(node.step.negative_prompt)

            if self.prompt is not None:
                outpaint_prompt_tokens.add(self.prompt, 2)  # Weighted
            if self.negative_prompt is not None:
                outpaint_negative_prompt_tokens.add(self.negative_prompt, 2)

            for i, image in enumerate(images):
                pipeline.controlnet = None
                result = pipeline(
                    image=image,
                    mask=outpaint_mask,
                    prompt=str(outpaint_prompt_tokens),
                    negative_prompt=str(outpaint_negative_prompt_tokens),
                    latent_callback=image_callback,
                    num_images_per_prompt=1,
                    **invocation_kwargs,
                )
                images[i] = result["images"][0]
                nsfw_content_detected[i] = nsfw_content_detected[i] or (
                    "nsfw_content_detected" in result and result["nsfw_content_detected"][0]
                )

        return images, nsfw_content_detected

    def get_serialization_dict(self) -> Dict[str, Any]:
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
                    "negative_prompt": self.upscale_diffusion_negative_prompt,
                    "controlnet": self.upscale_diffusion_controlnet,
                    "scale_chunking_size": self.upscale_diffusion_scale_chunking_size,
                    "scale_chunking_blur": self.upscale_diffusion_scale_chunking_blur,
                }
            upscale_dict = {
                "method": self.upscale,
                "amount": self.outscale,
                "iterative": self.upscale_iterative,
                "diffusion": upscale_diffusion_dict,
            }

        return {
            "model": self.model,
            "lora": self.lora,
            "inversion": self.inversion,
            "width": self.width,
            "height": self.height,
            "size": self.size,
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image": self.image,
            "image_callback_steps": self.image_callback_steps,
            "nodes": [node.get_serialization_dict() for node in self.nodes],
            "samples": self.samples,
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
            "nodes": [
                DiffusionNode.deserialize_dict(node_dict)
                for node_dict in plan_dict.get("nodes", [])
            ],
        }

        for arg in [
            "size",
            "lora",
            "inversion",
            "width",
            "height",
            "image_callback_steps",
            "image",
            "chunking_size",
            "chunking_blur",
            "samples",
            "seed",
            "prompt",
            "negative_prompt",
            "build_tensorrt",
        ]:
            if arg in plan_dict:
                kwargs[arg] = plan_dict[arg]

        upscale = plan_dict.get("upscale", None)
        if isinstance(upscale, dict):
            kwargs["upscale"] = upscale["method"]
            kwargs["outscale"] = upscale["amount"]
            kwargs["upscale_iterative"] = upscale.get("iterative", False)
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
                    "negative_prompt",
                    "scale_chunking_size",
                    "scale_chunking_blur",
                ]:
                    if arg in upscale_diffusion:
                        kwargs[f"upscale_diffusion_{arg}"] = upscale_diffusion[arg]
        result = DiffusionPlan(**kwargs)
        return result

    @staticmethod
    def create_mask(
        width: int, height: int, left: int, top: int, right: int, bottom: int
    ) -> PIL.Image.Image:
        """
        Creates a mask from 6 dimensions
        """
        image = PIL.Image.new("RGB", (width, height))
        draw = PIL.ImageDraw.Draw(image)
        draw.rectangle([(left, top), (right, bottom)], fill="#ffffff")
        return image

    @staticmethod
    def from_nodes(
        size: Optional[int] = DEFAULT_SIZE,
        model: Optional[str] = None,
        lora: Optional[
            Union[str, List[str], Tuple[str, float], List[Union[str, Tuple[str, float]]]]
        ] = None,
        inversion: Optional[Union[str, List[str]]] = None,
        model_prompt: Optional[str] = None,
        model_negative_prompt: Optional[str] = None,
        samples: int = 1,
        seed: Optional[int] = None,
        width: Optional[int] = DEFAULT_SIZE,
        height: Optional[int] = DEFAULT_SIZE,
        image_callback_steps: int = DEFAULT_IMAGE_CALLBACK_STEPS,
        nodes: List[NodeDict] = [],
        chunking_size: Optional[int] = DEFAULT_CHUNKING_SIZE,
        chunking_blur: Optional[int] = DEFAULT_CHUNKING_BLUR,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = DEFAULT_INFERENCE_STEPS,
        guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE,
        outscale: Optional[int] = 1,
        upscale: Optional[
            Union[
                Literal[
                    "esrgan",
                    "esrganime",
                    "gfpgan",
                    "lanczos",
                    "bilinear",
                    "bicubic",
                    "lanczos",
                    "nearest",
                ],
                List[
                    Literal[
                        "esrgan",
                        "esrganime",
                        "gfpgan",
                        "lanczos",
                        "bilinear",
                        "bicubic",
                        "lanczos",
                        "nearest",
                    ]
                ],
            ]
        ] = None,
        upscale_diffusion: bool = False,
        upscale_iterative: bool = False,
        upscale_diffusion_steps: Optional[Union[int, List[int]]] = DEFAULT_UPSCALE_DIFFUSION_STEPS,
        upscale_diffusion_guidance_scale: Optional[
            Union[float, int, List[Union[float, int]]]
        ] = DEFAULT_UPSCALE_DIFFUSION_GUIDANCE_SCALE,
        upscale_diffusion_strength: Optional[
            Union[float, List[float]]
        ] = DEFAULT_UPSCALE_DIFFUSION_STRENGTH,
        upscale_diffusion_prompt: Optional[Union[str, List[str]]] = DEFAULT_UPSCALE_PROMPT,
        upscale_diffusion_negative_prompt: Optional[
            Union[str, List[str]]
        ] = DEFAULT_UPSCALE_NEGATIVE_PROMPT,
        upscale_diffusion_controlnet: Optional[
            Union[
                Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"],
                List[Literal["canny", "tile", "mlsd", "hed", "scribble", "inpaint"]],
            ]
        ] = None,
        upscale_diffusion_chunking_size: Optional[int] = DEFAULT_UPSCALE_DIFFUSION_CHUNKING_SIZE,
        upscale_diffusion_chunking_blur: Optional[int] = DEFAULT_UPSCALE_DIFFUSION_CHUNKING_BLUR,
        upscale_diffusion_scale_chunking_size: bool = True,
        upscale_diffusion_scale_chunking_blur: bool = True,
    ) -> DiffusionPlan:
        """
        Assembles a diffusion plan from step dictionaries.
        """
        # First instantiate the plan
        plan = DiffusionPlan(
            model=model,
            lora=lora,
            inversion=inversion,
            samples=samples,
            size=size,
            seed=seed,
            width=width,
            height=height,
            image_callback_steps=image_callback_steps,
            upscale=upscale,
            outscale=outscale,
            prompt=prompt,
            chunking_size=chunking_size,
            chunking_blur=chunking_blur,
            negative_prompt=negative_prompt,
            upscale_iterative=upscale_iterative,
            upscale_diffusion=upscale_diffusion,
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
        upscale_diffusion_negative_prompt_tokens = [TokenMerger()]

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
        if prompt:
            for token_merger in upscale_diffusion_prompt_tokens:
                token_merger.add(prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if model_prompt:
            for token_merger in upscale_diffusion_prompt_tokens:
                token_merger.add(model_prompt, MODEL_PROMPT_WEIGHT)

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
        if negative_prompt:
            for token_merger in upscale_diffusion_negative_prompt_tokens:
                token_merger.add(negative_prompt, GLOBAL_PROMPT_UPSCALE_WEIGHT)
        if model_negative_prompt:
            for token_merger in upscale_diffusion_negative_prompt_tokens:
                token_merger.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)

        # Now assemble the diffusion steps
        node_count = len(nodes)
        if node_count == 0:
            # Basic txt2img
            prompt_tokens = TokenMerger()
            if prompt:
                prompt_tokens.add(prompt)
            if model_prompt:
                prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)

            negative_prompt_tokens = TokenMerger()
            if negative_prompt:
                negative_prompt_tokens.add(negative_prompt)
            if model_negative_prompt:
                negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)

            step = DiffusionStep(
                width=width,
                height=height,
                prompt=str(prompt_tokens),
                negative_prompt=str(negative_prompt_tokens),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            plan.nodes = [DiffusionNode([(0, 0), (plan.width, plan.height)], step)]

            plan.upscale_diffusion_prompt = [
                str(merger) for merger in upscale_diffusion_prompt_tokens
            ]
            plan.upscale_diffusion_negative_prompt = [
                str(merger) for merger in upscale_diffusion_negative_prompt_tokens
            ]
            return plan

        # Using the diffusion canvas, assemble a multi-step plan
        for i, node_dict in enumerate(nodes):
            step = DiffusionStep(
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale
            )

            node_width = int(node_dict["w"])
            node_height = int(node_dict["h"])
            node_left = int(node_dict["x"])
            node_top = int(node_dict["y"])
            node_fit = node_dict.get("fit", None)
            node_anchor = node_dict.get("anchor", None)
            node_bounds = [
                (node_left, node_top),
                (node_left + node_width, node_top + node_height),
            ]

            node_infer = node_dict.get("infer", False)
            node_inpaint = node_dict.get("inpaint", False)
            node_control = node_dict.get("control", False)

            node_controlnet = node_dict.get("controlnet", None)
            node_prompt = node_dict.get("prompt", None)
            node_negative_prompt = node_dict.get("negative_prompt", None)
            node_strength: Optional[float] = node_dict.get("strength", None)
            node_conditioning_scale: Optional[float] = node_dict.get("conditioning_scale", None)
            node_image = node_dict.get("image", None)
            node_inpaint_mask = node_dict.get("mask", None)
            node_process_control_image = node_dict.get("process_control_image", True)
            node_scale_to_model_size = node_dict.get("scale_to_model_size", True)
            node_remove_background = bool(node_dict.get("remove_background", False))
            node_inference_steps: Optional[int] = node_dict.get("inference_steps", None)  # type: ignore[assignment]
            node_guidance_scale: Optional[float] = node_dict.get("guidance_scale", None)  # type: ignore[assignment]

            node_prompt_tokens = TokenMerger()
            node_negative_prompt_tokens = TokenMerger()

            if node_prompt:
                node_prompt_tokens.add(node_prompt)
                for merger in upscale_diffusion_prompt_tokens:
                    merger.add(node_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if prompt and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_prompt_tokens.add(prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_prompt:
                node_prompt_tokens.add(model_prompt, MODEL_PROMPT_WEIGHT)

            if node_negative_prompt:
                node_negative_prompt_tokens.add(node_negative_prompt)
                for merger in upscale_diffusion_negative_prompt_tokens:
                    merger.add(node_negative_prompt, UPSCALE_PROMPT_STEP_WEIGHT / node_count)
            if negative_prompt and node_image:
                # Only add global prompt to image nodes, it overrides too much on region nodes
                node_negative_prompt_tokens.add(negative_prompt, GLOBAL_PROMPT_STEP_WEIGHT)
            if model_negative_prompt:
                node_negative_prompt_tokens.add(model_negative_prompt, MODEL_PROMPT_WEIGHT)

            if node_image:
                if node_remove_background:
                    node_image = remove_background(node_image)

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

                if node_inpaint and node_inpaint_mask:
                    # Inpaint prior to anything else. First invert mask
                    node_inpaint_mask = PIL.ImageOps.invert(node_inpaint_mask.convert("L"))
                    if node_image_needs_outpainting:
                        # Merge inpaint and outpaint masks
                        node_mask.paste(node_inpaint_mask)
                        node_inverse_alpha = PIL.Image.eval(node_alpha_clamp, lambda a: 255 - a)
                        node_mask.paste(white, mask=node_inverse_alpha)
                    else:
                        # Just use inpaint mask
                        node_mask.paste(node_inpaint_mask)

                    inpaint_image_step = DiffusionStep(
                        image=node_image,
                        mask=feather_mask(node_mask.convert("1")),
                        prompt=str(node_prompt_tokens),
                        negative_prompt=str(node_negative_prompt_tokens),
                        guidance_scale=guidance_scale,
                    )
                    step_image = inpaint_image_step
                elif node_image_needs_outpainting and (node_infer or node_control):
                    # There are gaps; add an outpaint step before the infer/control
                    outpaint_image_step = DiffusionStep(
                        image=node_image,
                        mask=feather_mask(node_mask.convert("1")),
                        prompt=str(node_prompt_tokens),
                        negative_prompt=str(node_negative_prompt_tokens),
                        guidance_scale=guidance_scale,
                    )
                    step_image = outpaint_image_step
                else:
                    step_image = node_image

                if not node_infer and not node_inpaint and not node_control:
                    # Just paste image
                    step.image = step_image
                elif not node_infer and node_inpaint and not node_control:
                    # Just inpaint image
                    step = step_image
                else:
                    # Some combination of ops
                    step.prompt = str(node_prompt_tokens)
                    step.negative_prompt = str(node_negative_prompt_tokens)
                    if node_strength:
                        step.strength = node_strength
                    if node_infer:
                        step.image = step_image
                    if node_control and node_controlnet:
                        step.controlnet = node_controlnet  # type: ignore
                        step.control_image = step_image
                        if not node_process_control_image:
                            step.process_control_image = False
                        if node_conditioning_scale:
                            step.conditioning_scale = node_conditioning_scale
            elif node_prompt:
                step.prompt = str(node_prompt_tokens)
                step.negative_prompt = str(node_negative_prompt_tokens)
                step.width = node_width
                step.height = node_height
                step.remove_background = node_remove_background
            else:
                raise ValueError("Can't assemble a node from arguments")

            # Set common args for node
            if node_inference_steps:
                step.num_inference_steps = node_inference_steps
            if node_guidance_scale:
                step.guidance_scale = node_guidance_scale
            if not node_scale_to_model_size:
                step.scale_to_model_size = False

            # Add step to plan
            plan.nodes.append(DiffusionNode(node_bounds, step))

        plan.upscale_diffusion_prompt = [str(prompt) for prompt in upscale_diffusion_prompt_tokens]
        plan.upscale_diffusion_negative_prompt = [
            str(negative_prompt) for negative_prompt in upscale_diffusion_negative_prompt_tokens
        ]
        return plan
