from __future__ import annotations
from contextlib import ExitStack
from datetime import datetime
from typing import Iterator, List, Optional, Callable, TYPE_CHECKING

from contextlib import contextmanager
from enfugue.diffusion.constants import MotionVectorPointDict
from enfugue.diffusion.support.model import SupportModel

if TYPE_CHECKING:
    from torch import device as Device, dtype as DType
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
    def __init__(self, network: DragNUWANet, device: Device, dtype: DType) -> None:
        self.network = network
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        image: Image,
        motion_vectors: List[List[MotionVectorPointDict]] = [],
        motion_bucket_id: int=27,
        batch_size: int=1,
        gaussian_sigma: int=20,
        noise_aug_strength: float=0.02,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        latent_callback: Optional[Callable[[Tensor], None]]=None,
        latent_callback_steps: Optional[int]=None,
    ) -> List[List[Image]]:
        """
        Executes the network
        """
        import torch
        from torchvision import transforms, utils
        from einops import repeat, rearrange
        from enfugue.diffusion.util.torch_util import motion_vector_conditioning_tensor
        condition = motion_vector_conditioning_tensor(
            width=self.network.args.width,
            height=self.network.args.height,
            frames=self.network.num_frames,
            gaussian_sigma=gaussian_sigma,
            motion_vectors=motion_vectors,
            device=self.device,
            dtype=self.dtype
        )

        input_condition = repeat(condition, "l h w c -> b l h w c", b=batch_size)
        input_condition = rearrange(input_condition, "b l h w c -> b l c h w", b=batch_size)

        image = image.resize((self.network.args.width, self.network.args.height)).convert("RGB")
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        input_image = repeat(transform_image(image), "c h w -> b c h w", b=batch_size).to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            with torch.autocast(self.device.type, dtype=self.dtype):
                conditioning_input = {
                    "cond_frames_without_noise": input_image,
                    "cond_frames": (input_image + noise_aug_strength * torch.randn_like(input_image)),
                    "motion_bucket_id": torch.tensor([motion_bucket_id]).to(device=self.device, dtype=self.dtype).repeat(batch_size * self.network.num_frames),
                    "fps_id": torch.tensor([self.network.args.fps]).to(device=self.device, dtype=self.dtype).repeat(batch_size * self.network.num_frames),
                    "cond_aug": torch.tensor([noise_aug_strength]).to(device=self.device, dtype=self.dtype).repeat(batch_size * self.network.num_frames),
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
                    conditional[key] = repeat(conditional[key], "b ... -> b t ...", t=self.network.num_frames)
                    conditional[key] = rearrange(conditional[key], "b t ... -> (b t) ...")
                    unconditional[key] = repeat(unconditional[key], "b ... -> b t ...", t=self.network.num_frames)
                    unconditional[key] = rearrange(unconditional[key], "b t ... -> (b t) ...")
                for key in ["vector"]:
                    conditional[key] = conditional[key].to(dtype=self.dtype)
                    unconditional[key] = unconditional[key].to(dtype=self.dtype)

                H, W = conditioning_input['cond_frames_without_noise'].shape[2:]
                shape = (self.network.num_frames, 4, H // 8, W // 8)
                randn = torch.randn(shape).to(device=self.device, dtype=self.dtype)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, self.network.num_frames).to(device=self.device, dtype=self.dtype)
                additional_model_inputs["num_video_frames"] = self.network.num_frames
                additional_model_inputs["flow"] = input_condition.repeat(2, 1, 1, 1, 1) # c and uc

                steps_complete = 0
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
                        progress_callback(steps_complete, self.network.args.sampler_config["num_steps"], 1 / step_duration_average)

                    if latent_callback is not None and latent_callback_steps is not None and steps_complete % latent_callback_steps == 0:
                        latent_callback(result)

                    step_time = current_time
                    return result

                samples_z = self.network.sampler(denoiser, randn, cond=conditional, uc=unconditional)
                samples = self.network.decode_first_stage(samples_z)
                samples = rearrange(samples, '(b l) c h w -> b l c h w', b=batch_size)
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
                return frame_sequences

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
        motion_bucket_id: int=27,
        batch_size: int=1,
        gaussian_sigma: int=20,
        noise_aug_strength: float=0.02,
        progress_callback: Optional[Callable[[int, int, float], None]]=None,
        latent_callback: Optional[Callable[[Tensor], None]]=None,
        latent_callback_steps: Optional[int]=None,
    ) -> List[List[Image]]:
        """
        Pass through to the processor
        """
        # Set args
        self.processor.network.seed = seed
        self.processor.network.args.width = width
        self.processor.network.args.height = height
        self.processor.network.args.fps = fps
        self.processor.network.args.num_frames = num_frames
        # Call
        return self.processor(
            image=image,
            motion_vectors=motion_vectors,
            motion_bucket_id=motion_bucket_id,
            batch_size=batch_size,
            gaussian_sigma=gaussian_sigma,
            noise_aug_strength=noise_aug_strength,
            progress_callback=progress_callback,
            latent_callback=latent_callback,
            latent_callback_steps=latent_callback_steps
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
    WEIGHTS_PATH = "https://huggingface.co/yinsming/DragNUWA/resolve/main/drag_nuwa_svd.pth"

    def nuwa(self, **kwargs) -> DragAnimatorPipeline:
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
        from enfugue.diffusion.animate.dragnuwa.net import DragNUWANet # type: ignore
        from enfugue.diffusion.animate.dragnuwa.utils import ( # type: ignore
            adaptively_load_state_dict
        )
        from enfugue.diffusion.util import load_state_dict
        weights_file = self.get_model_file(self.WEIGHTS_PATH)
        with self.context():
            network = DragNUWANet(**kwargs)
            network.eval()
            network.to(device=self.device)
            adaptively_load_state_dict(
                network,
                load_state_dict(weights_file, self.device.type)
            )
            network.to(dtype=self.dtype)
            processor = DragAnimatorProcessor(
                network=network,
                device=self.device,
                dtype=self.dtype
            )
            yield processor
            del processor
            del network
