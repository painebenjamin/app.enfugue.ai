from __future__ import annotations

from math import ceil
from contextlib import ExitStack
from datetime import datetime
from typing import Iterator, List, Optional, Callable, Any, Dict, TYPE_CHECKING

from contextlib import contextmanager
from enfugue.util import logger
from enfugue.diffusion.constants import *
from enfugue.diffusion.support.model import SupportModel

if TYPE_CHECKING:
    from torch import device as Device, dtype as DType, Tensor
    from PIL.Image import Image
    from enfugue.diffusion.animate.dragnuwa.net import DragNUWANet # type: ignore

__all__ = [
    "DragAnimatorProcessor",
    "DragAnimatorPipeline",
    "DragAnimator"
]

class DragAnimatorProcessor:
    """
    Calls DragNUWA
    """
    def __init__(
        self,
        root_dir: str,
        model_dir: str,
        network: DragNUWANet,
        device: Device,
        dtype: DType
    ) -> None:
        self.root_dir = root_dir
        self.model_dir = model_dir
        self.network = network
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        image: Image,
        width: int=576,
        height: int=320,
        fps: int=4,
        seed: int=42,
        motion_bucket_id: int=27,
        batch_size: int=1,
        gaussian_sigma: int=20,
        noise_aug_strength: float=0.02,
        min_guidance_scale: float=1.0,
        max_guidance_scale: float=3.0,
        num_frames: int=14,
        frame_window_size: int=14,
        num_inference_steps: int=25,
        optical_flow: Optional[Dict[OPTICAL_FLOW_METHOD_LITERAL, List[List[Image]]]] = None,
        motion_vectors: List[List[MotionVectorPointDict]] = [],
        motion_vector_repeat_window: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        latent_callback: Optional[Callable[[Tensor, int], None]]=None,
        latent_callback_steps: Optional[int]=None,
    ) -> List[List[Image]]:
        """
        Executes the network
        """
        import torch
        import numpy as np
        from torchvision import transforms, utils
        from PIL import Image
        from einops import repeat, rearrange
        from enfugue.diffusion.util import (
            Video,
            motion_vector_conditioning_tensor,
            optical_flow_conditioning_tensor,
        )

        # Set args
        self.network.args.width = width
        self.network.args.height = height
        self.network.args.fps = fps

        # Set sampler
        self.network.sampler.num_steps = num_inference_steps

        num_iterations = ceil(num_frames / frame_window_size)
        num_zero_frames = num_iterations - 1
        # num_zero_frames = 0

        # Build condition
        condition_frames = frame_window_size if motion_vector_repeat_window else num_frames-num_zero_frames
        condition = motion_vector_conditioning_tensor(
            width=width,
            height=height,
            frames=condition_frames,
            gaussian_sigma=gaussian_sigma,
            motion_vectors=motion_vectors,
            device=self.device,
            dtype=self.dtype
        )
        if motion_vector_repeat_window:
            # Repeat condition
            condition = condition.repeat(num_iterations, 1, 1, 1)[:num_frames-num_zero_frames, :, :, :]

        # Calculate optical flow and add to condition
        if optical_flow:
            for optical_flow_method in optical_flow:
                if optical_flow_method == "unimatch":
                    from enfugue.diffusion.support.unimatch import Unimatch
                    with Unimatch(
                        root_dir=self.root_dir,
                        model_dir=self.model_dir,
                        device=self.device,
                        dtype=self.dtype
                    ).flow() as get_flow:
                        for image_sequence in optical_flow[optical_flow_method]:
                            image_sequence = image_sequence[:condition_frames]
                            sequence_condition = optical_flow_conditioning_tensor(
                                np.array([
                                    get_flow(
                                        image_sequence[i],
                                        image_sequence[i+1]
                                    )
                                    for i in range(len(image_sequence)-1)
                                ]),
                                device=self.device,
                                dtype=self.dtype
                            )
                            condition[0:sequence_condition.shape[0]] += sequence_condition[:condition.shape[0]]
                else:
                    for image_sequence in optical_flow[optical_flow_method]:
                        image_sequence = image_sequence[:condition_frames]
                        sequence_gaussian_sigma: Optional[int] = None
                        if optical_flow_method == "lucas-kanade":
                            flow = Video(image_sequence).sparse_flow()
                            sequence_gaussian_sigma = gaussian_sigma
                        else:
                            flow = Video(image_sequence).dense_flow(optical_flow_method)
                        sequence_condition = optical_flow_conditioning_tensor(
                            np.array([flow_frame for flow_frame in flow]),
                            gaussian_sigma=sequence_gaussian_sigma,
                            device=self.device,
                            dtype=self.dtype
                        )
                        condition[0:sequence_condition.shape[0]] += sequence_condition[:condition.shape[0]]

        # Insert zero-frames
        for i in range(num_zero_frames):
            index = (i + 1) * frame_window_size
            condition = torch.cat([
                condition[:index, :, :, :],
                torch.zeros_like(condition[0:1, :, :, :]),
                condition[index:, :, :, :]
            ], dim=0)

        input_condition = repeat(condition, "l h w c -> b l h w c", b=batch_size)
        input_condition = rearrange(input_condition, "b l h w c -> b l c h w", b=batch_size)

        image = image.resize((width, height)).convert("RGB")
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        output_sequences: Optional[List[List[Image]]] = None
        input_image = repeat(transform_image(image), "c h w -> b c h w", b=batch_size).to(device=self.device, dtype=self.dtype)

        steps_complete = 0
        steps_total = num_iterations * num_inference_steps

        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self.dtype):
                for i in range(num_iterations):
                    # Reset the seed every iteration
                    self.network.seed = seed

                    iteration_frame_start = frame_window_size * i
                    iteration_frame_end = frame_window_size * (i + 1)
                    if iteration_frame_end > num_frames:
                        iteration_frame_end = num_frames

                    num_iteration_frames = iteration_frame_end - iteration_frame_start
                    self.network.sampler.guider.rescale(
                        min_scale=min_guidance_scale,
                        max_scale=max_guidance_scale,
                        num_frames=num_iteration_frames,
                    )
                    conditioning_input = {
                        "cond_frames_without_noise": input_image,
                        "cond_frames": (input_image + noise_aug_strength * torch.randn_like(input_image)),
                        "motion_bucket_id": torch.tensor([motion_bucket_id]).to(device=self.device, dtype=self.dtype).repeat(batch_size * num_iteration_frames),
                        "fps_id": torch.tensor([fps]).to(device=self.device, dtype=self.dtype).repeat(batch_size * num_iteration_frames),
                        "cond_aug": torch.tensor([noise_aug_strength]).to(device=self.device, dtype=self.dtype).repeat(batch_size * num_iteration_frames),
                    }
                    batch_uc = dict([
                        (key, value.clone())
                        for key, value in conditioning_input.items()
                        if isinstance(value, torch.Tensor)
                    ])
                    conditional, unconditional = self.network.conditioner.get_unconditional_conditioning(
                        conditioning_input,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"]
                    )
                    for key in ["crossattn", "concat"]:
                        conditional[key] = repeat(conditional[key], "b ... -> b t ...", t=num_iteration_frames)
                        conditional[key] = rearrange(conditional[key], "b t ... -> (b t) ...")
                        unconditional[key] = repeat(unconditional[key], "b ... -> b t ...", t=num_iteration_frames)
                        unconditional[key] = rearrange(unconditional[key], "b t ... -> (b t) ...")
                    for key in ["vector"]:
                        conditional[key] = conditional[key].to(dtype=self.dtype)
                        unconditional[key] = unconditional[key].to(dtype=self.dtype)

                    H, W = conditioning_input["cond_frames_without_noise"].shape[2:]
                    shape = (num_iteration_frames, 4, H // 8, W // 8)
                    randn = torch.randn(shape).to(device=self.device, dtype=self.dtype)

                    additional_model_inputs: Dict[str, Any] = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_iteration_frames).to(device=self.device, dtype=self.dtype)
                    additional_model_inputs["num_video_frames"] = num_iteration_frames
                    flow_condition = input_condition[:, iteration_frame_start:iteration_frame_end, :, :, :]
                    flow_condition = flow_condition.repeat(2, 1, 1, 1, 1) # c and uc
                    additional_model_inputs["flow"] = flow_condition

                    step_time = datetime.now()
                    step_times: List[float] = []

                    def denoiser(input: Tensor, sigma: Tensor, c: Tensor) -> Tensor:
                        nonlocal steps_complete, step_time
                        result = self.network.denoiser(self.network.model, input.to(self.dtype), sigma, c, **additional_model_inputs)
                        steps_complete += 1
                        current_time = datetime.now()

                        if progress_callback is not None:
                            duration = (current_time - step_time).total_seconds()
                            step_times.append(duration)
                            step_time_slice = step_times[-5:]
                            step_duration_average = sum(step_time_slice) / len(step_time_slice)
                            progress_callback(steps_complete, steps_total, 1 / step_duration_average)

                        if latent_callback is not None and latent_callback_steps and steps_complete % latent_callback_steps == 0:
                            latent_callback(result, i)

                        step_time = current_time
                        return result

                    samples_z = self.network.sampler(denoiser, randn, cond=conditional, uc=unconditional)
                    samples = self.network.decode_first_stage(samples_z)
                    samples = rearrange(samples, "(b l) c h w -> b l c h w", b=batch_size)
                    to_pil = transforms.ToPILImage("RGB")
                    frame_sequences = [
                        [
                            to_pil(
                                utils.make_grid(
                                    frame.to(torch.float32).cpu(),
                                    normalize=True,
                                    value_range=(-1, 1)
                                )
                            )
                            for frame in batch
                        ]
                        for batch in samples
                    ]
                    if output_sequences is None:
                        output_sequences = frame_sequences
                    else:
                        for i, sequence in enumerate(frame_sequences):
                            # Merge last frame of previous sequence and first frame of this one
                            output_sequences[i][-1] = Image.blend(output_sequences[i][-1], sequence[0], 0.5)
                            output_sequences[i].extend(sequence[1:])
                    input_image = repeat(transform_image(output_sequences[-1][-1]), "c h w -> b c h w", b=batch_size).to(device=self.device, dtype=self.dtype)
                if output_sequences is None:
                    raise IOError("No results!")
                return output_sequences

class DragAnimatorPipeline:
    """
    This class functions as a re-callable pipeline that can be deleted
    """
    def __init__(self, stack: ExitStack, processor: DragAnimatorProcessor):
        self.stack = stack
        self.processor = processor

    def __call__(
        self,
        image: Image,
        motion_vectors: List[List[MotionVectorPointDict]] = [],
        width: int=576,
        height: int=320,
        fps: int=4,
        seed: int=42,
        num_frames: int=14,
        frame_window_size: int=14,
        motion_bucket_id: int=27,
        batch_size: int=1,
        gaussian_sigma: int=20,
        noise_aug_strength: float=0.02,
        num_inference_steps: int=25,
        min_guidance_scale: float=1.0,
        max_guidance_scale: float=3.0,
        optical_flow: Optional[Dict[OPTICAL_FLOW_METHOD_LITERAL, List[List[Image]]]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        motion_vector_repeat_window: bool=False,
        latent_callback: Optional[Callable[[Tensor, int], None]]=None,
        latent_callback_steps: Optional[int]=None,
    ) -> List[List[Image]]:
        """
        Pass through to the processor
        """
        # Call
        return self.processor(
            fps=fps,
            image=image,
            seed=seed,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_window_size=frame_window_size,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            optical_flow=optical_flow,
            motion_vectors=motion_vectors,
            motion_bucket_id=motion_bucket_id,
            batch_size=batch_size,
            gaussian_sigma=gaussian_sigma,
            noise_aug_strength=noise_aug_strength,
            progress_callback=progress_callback,
            latent_callback=latent_callback,
            latent_callback_steps=latent_callback_steps,
            motion_vector_repeat_window=motion_vector_repeat_window
        )

    def __del__(self) -> None:
        """
        Exit the stack when deleted
        """
        self.stack.close()

class DragAnimator(SupportModel):
    """
    Maps DragNUWA to a support model and returns an easy callable
    """
    WEIGHTS_PATH = "https://huggingface.co/benjamin-paine/dragnuwa-pruned-safetensors/resolve/main/dragnuwa-svd-pruned.safetensors"
    WEIGHTS_PATH_FP16 = "https://huggingface.co/benjamin-paine/dragnuwa-pruned-safetensors/resolve/main/dragnuwa-svd-pruned.fp16.safetensors"

    def nuwa(self, **kwargs: Any) -> DragAnimatorPipeline:
        """
        Gets a re-usable dragnuwa model
        """
        stack = ExitStack()
        processor = stack.enter_context(self.nuwa_processor(**kwargs))
        return DragAnimatorPipeline(stack, processor)

    @contextmanager
    def nuwa_processor(self, **kwargs: Any) -> Iterator[DragAnimatorProcessor]:
        """
        Gets the DragNUWA model and yields a processor
        """
        import torch
        from enfugue.diffusion.animate.dragnuwa.net import DragNUWANet # type: ignore
        from enfugue.diffusion.animate.dragnuwa.utils import ( # type: ignore
            adaptively_load_state_dict
        )
        weights_file = self.get_model_file(self.WEIGHTS_PATH_FP16 if self.dtype is torch.float16 else self.WEIGHTS_PATH)
        with self.context():
            network = DragNUWANet(device=self.device.type, **kwargs)
            network.eval()
            network.to(device=self.device, dtype=self.dtype)
            logger.debug(f"Loading DragNUWA state dictionary from {weights_file}")
            adaptively_load_state_dict(
                network,
                weights_file,
                device=self.device.type,
                dtype=self.dtype
            )
            processor = DragAnimatorProcessor(
                root_dir=self.root_dir,
                model_dir=self.model_dir,
                network=network,
                device=self.device,
                dtype=self.dtype
            )
            yield processor
            del processor
            del network
