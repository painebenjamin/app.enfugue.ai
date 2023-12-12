from __future__ import annotations

import inspect

from dataclasses import dataclass, asdict

from typing import List, Optional, Callable, Dict, Any, Literal, Union, Tuple, TYPE_CHECKING

from enfugue.util import logger, redact_images_from_metadata

from PIL import Image

if TYPE_CHECKING:
    from enfugue.diffusion.manager import DiffusionPipelineManager
    from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

__all__ = ["StableVideoDiffusionInvocation"]

@dataclass
class StableVideoDiffusionInvocation:
    """
    A serializable class holding all vars for executing SVD
    """
    image: Image.Image # Required
    model: Literal["svd", "svd_xt"]="svd"
    num_inference_steps: int=25
    min_guidance_scale: float=1.0
    max_guidance_scale: float=3.0
    fps: int=7
    noise_aug_strength: float=0.02
    decode_chunk_size: Optional[int]=1
    motion_bucket_id: int=127
    reflect: bool=False
    interpolate_frames: Optional[Union[int, Tuple[int, ...], List[int]]]=None

    @property
    def animation_frames(self) -> int:
        """
        Returns the number of animation frames
        """
        return 25 if self.model == "svd_xt" else 18

    @property
    def kwargs(self) -> Dict[str, Any]:
        """
        Returns the arguments to pass to `svd_img2vid`
        """
        return {
            "image": self.image,
            "use_xt": self.model == "svd_xt",
            "decode_chunk_size": self.decode_chunk_size,
            "num_inference_steps": self.num_inference_steps,
            "min_guidance_scale": self.min_guidance_scale,
            "max_guidance_scale": self.max_guidance_scale,
            "fps": self.fps,
            "noise_aug_strength": self.noise_aug_strength,
            "motion_bucket_id": self.motion_bucket_id
        }

    @classmethod
    def assemble(cls, **kwargs: Any) -> StableVideoDiffusionInvocation:
        """
        Assembles from a payload, strips arguments
        """
        accepted_parameters = inspect.signature(cls).parameters
        accepted_kwargs = dict([
            (k, v) for k, v in kwargs.items()
            if k in accepted_parameters
        ])
        ignored_kwargs = set(list(kwargs.keys())) - set(list(accepted_parameters))
        if ignored_kwargs:
            logger.warning(f"Ignored keyword arguments {ignored_kwargs}")
        return cls(**accepted_kwargs)

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        image_callback: Optional[Callable[[List[Image.Image]], None]] = None,
        image_callback_steps: Optional[int] = None,
    ) -> StableDiffusionPipelineOutput:
        """
        This is the main interface for execution.
        """
        frames = pipeline.svd_img2vid(
            task_callback=task_callback,
            progress_callback=progress_callback,
            image_callback=image_callback,
            image_callback_steps=image_callback_steps,
            **self.kwargs
        )
        from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
        return StableDiffusionPipelineOutput(
            images=frames,
            nsfw_content_detected=[False]*len(frames)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        Returns the invocation as a dict
        """
        return asdict(self)

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Returns invocation metadata as a dict
        """
        metadata = self.serialize()
        redact_images_from_metadata(metadata)
        return metadata
