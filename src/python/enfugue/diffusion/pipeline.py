from typing import (
    Any,
    List,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterator,
    Literal,
    Mapping,
    cast,
)

import PIL
import PIL.Image
import math
import torch
import safetensors.torch
import datetime
import numpy as np

from contextlib import contextmanager
from collections import defaultdict

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
)
from diffusers.schedulers import KarrasDiffusionSchedulers, DDIMScheduler

from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel

from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)

from diffusers.utils import randn_tensor, PIL_INTERPOLATION
from diffusers.image_processor import VaeImageProcessor

from enfugue.util import logger

# This is ~64k×64k. Absurd, but I don't judge
PIL.Image.MAX_IMAGE_PIXELS = 2**32


class EnfugueStableDiffusionPipeline(StableDiffusionPipeline):
    """
    This pipeline merges all of the following:
    1. txt2img
    2. img2img
    3. inpainting/outpainting
    4. controlnet
    5. tensorrt
    """

    controlnet: Optional[ControlNetModel]
    scheduler: DDIMScheduler  # Must use DDIM for multi-diffusion

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
        super(EnfugueStableDiffusionPipeline, self).__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,
        )
        # Override scheduler to DDIM for multidiffusion
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)

        # Enfugue engine settings
        self.engine_size = engine_size
        self.chunking_size = chunking_size
        self.chunking_blur = chunking_blur

        # Hide tqdm
        self.set_progress_bar_config(disable=True)

        # Add an image processor for later
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # Register controlnet, it can be None
        self.register_modules(controlnet=controlnet)

    @contextmanager
    def get_runtime_context(
        self, batch_size: int, device: Union[str, torch.device]
    ) -> Iterator[None]:
        """
        Used by other implementations (tensorrt), but not base.
        """
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cpu":
            with torch.autocast("cpu"):
                yield
        else:
            yield

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        multiplier: int = 1,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """
        Fix adapted from here: https://github.com/huggingface/diffusers/issues/3064#issuecomment-1545013909
        """
        if not isinstance(
            pretrained_model_name_or_path_or_dict, str
        ) or not pretrained_model_name_or_path_or_dict.endswith(".safetensors"):
            return super(EnfugueStableDiffusionPipeline, self).load_lora_weights(
                pretrained_model_name_or_path_or_dict, **kwargs
            )

        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"

        # load LoRA weight from .safetensors
        state_dict = safetensors.torch.load_file(
            pretrained_model_name_or_path_or_dict, device="cpu"
        )

        updates: Mapping[str, Any] = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split(".", 1)
            updates[layer][elem] = value

        index = 0
        # directly update weight in diffusers model
        for layer, elems in updates.items():
            index += 1

            if "text" in layer:
                layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                curr_layer = self.text_encoder
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = self.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems["lora_up.weight"].to(dtype)
            weight_down = elems["lora_down.weight"].to(dtype)
            alpha = elems["alpha"]
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += (
                    multiplier
                    * alpha
                    * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2))
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Denomalizes image data from [-1, 1] to [0, 1]
        """
        return (latents / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def prepare_mask_and_image(
        self,
        mask: Union[np.ndarray, PIL.Image.Image, torch.Tensor],
        image: Union[np.ndarray, PIL.Image.Image, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a mask and image for inpainting.
        """
        if isinstance(image, torch.Tensor):
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            # Batch and add channel dim for single mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

            # Batch single mask or add channel dim
            if mask.ndim == 3:
                # Single batched mask, no channel dim or single mask not batched but channel dim
                if mask.shape[0] == 1:
                    mask = mask.unsqueeze(0)

                # Batched masks no channel dim
                else:
                    mask = mask.unsqueeze(1)

            assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
            assert (
                image.shape[-2:] == mask.shape[-2:]
            ), "Image and Mask must have the same spatial dimensions"
            assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")

            # Check mask is in [0, 1]
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")

            # Binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            # Image as float32
            image = image.to(dtype=torch.float32)
        elif isinstance(mask, torch.Tensor):
            raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
        else:
            # preprocess image
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

            # preprocess mask
            if isinstance(mask, (PIL.Image.Image, np.ndarray)):
                mask = [mask]

            if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
                mask = np.concatenate(
                    [np.array(m.convert("L"))[None, None, :] for m in mask], axis=0
                )
                mask = mask.astype(np.float32) / 255.0
            elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
                mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        return mask, masked_image

    @torch.no_grad()
    def create_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Creates random latents of a particular shape and type.
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        logger.debug(f"Creating random latents of shape {shape} and type {dtype}")
        random_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return random_latents * self.scheduler.init_noise_sigma

    @torch.no_grad()
    def encode_image_unchunked(
        self, image: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Encodes an image without chunking using the VAE.
        """
        return self.vae.encode(image).latent_dist.sample(generator) * self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Encodes an image in chunks using the VAE.
        """
        _, _, height, width = image.shape
        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        if total_steps == 1:
            result = self.encode_image_unchunked(image, generator)
            if progress_callback is not None:
                progress_callback()
            return result

        logger.debug(f"Encoding image in {total_steps} steps.")

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        engine_latent_size = self.engine_size // self.vae_scale_factor
        num_channels = self.vae.config.latent_channels

        count = torch.zeros((1, num_channels, latent_height, latent_width)).to(device=device)
        value = torch.zeros_like(count)

        for i, (top, bottom, left, right) in enumerate(chunks):
            top_px = top * self.vae_scale_factor
            bottom_px = bottom * self.vae_scale_factor
            left_px = left * self.vae_scale_factor
            right_px = right * self.vae_scale_factor

            image_view = image[:, :, top_px:bottom_px, left_px:right_px]

            encoded_image = self.vae.encode(image_view).latent_dist.sample(generator).to(device)

            # Blur edges as needed
            multiplier = torch.ones_like(encoded_image)
            if self.chunking_blur > 0:
                chunking_blur_latent_size = self.chunking_blur // self.vae_scale_factor
                blur_left = left > 0
                blur_top = top > 0
                blur_right = right < latent_width
                blur_bottom = bottom < latent_height

                for j in range(chunking_blur_latent_size):
                    mult = (j + 1) / (chunking_blur_latent_size + 1)
                    if blur_left:
                        multiplier[:, :, :, j] *= mult
                    if blur_top:
                        multiplier[:, :, j, :] *= mult
                    if blur_right:
                        multiplier[:, :, :, engine_latent_size - j - 1] *= mult
                    if blur_bottom:
                        multiplier[:, :, engine_latent_size - j - 1, :] *= mult

            value[:, :, top:bottom, left:right] += encoded_image * multiplier
            count[:, :, top:bottom, left:right] += multiplier

            if progress_callback is not None:
                progress_callback()

        return torch.where(count > 0, value / count, value) * self.vae.config.scaling_factor

    @torch.no_grad()
    def prepare_image_latents(
        self,
        image: Union[torch.Tensor, PIL.Image.Image],
        timestep: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Prepares latents from an image, adding initial noise for img2img inference
        """
        image = image.to(device=device, dtype=dtype)
        init_latents = self.encode_image(
            image, device=device, generator=generator, progress_callback=progress_callback
        ).to(dtype=dtype)

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # duplicate images to match batch size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # add noise in accordance with timesteps
        return self.scheduler.add_noise(init_latents, noise, timestep)

    @torch.no_grad()
    def prepare_mask_latents(
        self,
        mask: Union[PIL.Image.Image, torch.Tensor],
        image: Union[PIL.Image.Image, torch.Tensor],
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator] = None,
        do_classifier_free_guidance: bool = False,
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares both mask and image latents for inpainting
        """
        tensor_height = height // self.vae_scale_factor
        tensor_width = width // self.vae_scale_factor
        tensor_size = (tensor_height, tensor_width)
        mask = torch.nn.functional.interpolate(mask, size=tensor_size)
        mask = mask.to(device=device, dtype=dtype)
        image = image.to(device=device, dtype=dtype)

        latents = self.encode_image(
            image, device=device, generator=generator, progress_callback=progress_callback
        ).to(device=device)

        # duplicate mask and latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if latents.shape[0] < batch_size:
            if not batch_size % latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            latents = latents.repeat(batch_size // latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        latents = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # aligning device to prevent device errors when concating it with the latent model input
        latents = latents.to(device=device, dtype=dtype)
        return mask, latents

    @torch.no_grad()
    def get_timesteps(
        self, num_inference_steps: int, strength: float, device: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Gets the original timesteps from the scheduler based on strength when doing img2img
        """
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def predict_noise_residual(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        embeddings: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Runs the UNet to predict noise residual.
        """
        return self.unet(
            latents,
            timestep,
            encoder_hidden_states=embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def prepare_control_image(
        self,
        image: Union[torch.Tensor, PIL.Image.Image, List[PIL.Image.Image]],
        width: int,
        height: int,
        batch_size: int,
        num_images_per_prompt: int,
        device: Union[str, torch.Tensor],
        dtype: torch.dtype,
        do_classifier_free_guidance=False,
    ):
        """
        Prepares an image for controlnet conditioning.
        """
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image.to(device=device, dtype=dtype)

    @torch.no_grad()
    def prepare_controlnet_inpaint_control_image(
        self,
        image: PIL.Image.Image,
        mask: PIL.Image.Image,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Combines the image and mask into a condition for controlnet inpainting.
        """
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask = np.array(mask.convert("L")).astype(np.float32) / 255.0

        assert (
            image.shape[0:1] == mask.shape[0:1]
        ), "image and image_mask must have the same image size"
        image[mask > 0.5] = -1.0  # set as masked pixel

        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        return image.to(device=device, dtype=dtype)

    def get_minimal_chunks(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Gets the minimum chunks that cover the shape in multiples of the engine width
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        latent_window_size = self.engine_size // self.vae_scale_factor
        horizontal_blocks = math.ceil(latent_width / latent_window_size)
        vertical_blocks = math.ceil(latent_height / latent_window_size)
        total_blocks = vertical_blocks * horizontal_blocks
        chunks = []

        for i in range(total_blocks):
            top = (i // horizontal_blocks) * latent_window_size
            bottom = top + latent_window_size

            left = (i % horizontal_blocks) * latent_window_size
            right = left + latent_window_size

            if bottom > latent_height:
                offset = bottom - latent_height
                bottom -= offset
                top -= offset
            if right > latent_width:
                offset = right - latent_width
                right -= offset
                left -= offset
            chunks.append((top, bottom, left, right))
        return chunks

    def get_chunks(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """
        Gets the chunked latent indices for multidiffusion
        """
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        if not self.chunking_size:
            return [(0, latent_height, 0, latent_width)]

        latent_chunking_size = self.chunking_size // self.vae_scale_factor
        latent_window_size = self.engine_size // self.vae_scale_factor

        vertical_blocks = math.ceil((latent_height - latent_window_size) / latent_chunking_size + 1)
        horizontal_blocks = math.ceil(
            (latent_width - latent_window_size) / latent_chunking_size + 1
        )
        total_blocks = vertical_blocks * horizontal_blocks
        chunks = []

        for i in range(int(total_blocks)):
            top = (i // horizontal_blocks) * latent_chunking_size
            bottom = top + latent_window_size
            left = (i % horizontal_blocks) * latent_chunking_size
            right = left + latent_window_size

            if bottom > latent_height:
                offset = bottom - latent_height
                bottom -= offset
                top -= offset
            if right > latent_width:
                offset = right - latent_width
                right -= offset
                left -= offset

            chunks.append((top, bottom, left, right))

        return chunks

    def get_controlnet_conditioning_blocks(
        self,
        device: Union[str, torch.device],
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.Tensor],
        conditioning_scale: float,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """
        Executes the controlnet
        """
        if controlnet_cond is None or self.controlnet is None:
            return None, None

        return self.controlnet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            return_dict=False,
        )

    @torch.no_grad()
    def denoise_unchunked(
        self,
        height: int,
        width: int,
        device: Union[str, torch.device],
        num_inference_steps: int,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        encoded_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        latent_callback: Optional[
            Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]
        ] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        conditioning_scale: float = 1.0,
        extra_step_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Executes the denoising loop without chunking.
        """
        if extra_step_kwargs is None:
            extra_step_kwargs = {}

        num_steps = len(timesteps)
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order
        logger.debug(f"Denoising image in {num_steps} steps (unchunked)")

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Get controlnet input if configured
            if control_image is not None:
                down_block, mid_block = self.get_controlnet_conditioning_blocks(
                    device=device,
                    latents=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoded_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=conditioning_scale,
                )
            else:
                down_block, mid_block = None, None

            # add other dimensions to unet input if set
            if mask is not None and mask_image is not None:
                latent_model_input = torch.cat(
                    [latent_model_input, mask, mask_image],
                    dim=1,
                )

            # predict the noise residual
            noise_pred = self.predict_noise_residual(
                latent_model_input,
                t,
                encoded_prompt_embeds,
                cross_attention_kwargs,
                down_block,
                mid_block,
            )

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Compute previous noisy sample
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
            ).prev_sample

            if progress_callback is not None:
                progress_callback()

            # call the callback, if provided
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and i % latent_callback_steps == 0
                and (
                    i == num_steps - 1
                    or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)
                )
            ):
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latents(
                        latent_callback_value, device=device, progress_callback=progress_callback
                    )
                    latent_callback_value = self.denormalize_latents(latent_callback_value)
                    if latent_callback_type != "pt":
                        latent_callback_value = self.image_processor.pt_to_numpy(
                            latent_callback_value
                        )
                        if latent_callback_type == "pil":
                            latent_callback_value = self.image_processor.numpy_to_pil(
                                latent_callback_value
                            )
                latent_callback(latent_callback_value)

        return latents

    @torch.no_grad()
    def denoise(
        self,
        height: int,
        width: int,
        device: Union[str, torch.device],
        num_inference_steps: int,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        encoded_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool = False,
        mask: Optional[torch.Tensor] = None,
        mask_image: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        progress_callback: Optional[Callable[[], None]] = None,
        latent_callback: Optional[
            Callable[[Union[torch.Tensor, np.ndarray, List[PIL.Image.Image]]], None]
        ] = None,
        latent_callback_steps: Optional[int] = 1,
        latent_callback_type: Literal["latent", "pt", "np", "pil"] = "latent",
        conditioning_scale: float = 1.0,
        extra_step_kwargs: Optional[Dict[str, Any]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Executes the denoising loop.
        """
        if extra_step_kwargs is None:
            extra_step_kwargs = {}

        chunks = self.get_chunks(height, width)
        num_chunks = len(chunks)

        if num_chunks == 1:
            return self.denoise_unchunked(
                height=height,
                width=width,
                device=device,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                latents=latents,
                encoded_prompt_embeds=encoded_prompt_embeds,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mask=mask,
                mask_image=mask_image,
                control_image=control_image,
                conditioning_scale=conditioning_scale,
                progress_callback=progress_callback,
                latent_callback=latent_callback,
                latent_callback_steps=latent_callback_steps,
                latent_callback_type=latent_callback_type,
                extra_step_kwargs=extra_step_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        num_steps = len(timesteps)
        num_warmup_steps = num_steps - num_inference_steps * self.scheduler.order

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor
        engine_latent_size = self.engine_size // self.vae_scale_factor

        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        total_num_steps = num_steps * num_chunks
        logger.debug(
            f"Denoising image in {total_num_steps} total steps ({num_inference_steps} inference steps * {num_chunks} chunks)"
        )

        for i, t in enumerate(timesteps):
            # zero view latents
            count.zero_()
            value.zero_()

            # iterate over chunks
            for j, (top, bottom, left, right) in enumerate(chunks):
                # Get pixel indices
                top_px = top * self.vae_scale_factor
                bottom_px = bottom * self.vae_scale_factor
                left_px = left * self.vae_scale_factor
                right_px = right * self.vae_scale_factor

                # Slice latents
                latents_for_view = latents[:, :, top:bottom, left:right]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents_for_view] * 2)
                    if do_classifier_free_guidance
                    else latents_for_view
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Get controlnet input if configured
                if control_image is not None:
                    down_block, mid_block = self.get_controlnet_conditioning_blocks(
                        device=device,
                        latents=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=encoded_prompt_embeds,
                        controlnet_cond=control_image[:, :, top_px:bottom_px, left_px:right_px],
                        conditioning_scale=conditioning_scale,
                    )
                else:
                    down_block, mid_block = None, None

                # add other dimensions to unet input if set
                if mask is not None and mask_image is not None:
                    latent_model_input = torch.cat(
                        [
                            latent_model_input,
                            mask[:, :, top:bottom, left:right],
                            mask_image[:, :, top:bottom, left:right],
                        ],
                        dim=1,
                    )

                # predict the noise residual
                noise_pred = self.predict_noise_residual(
                    latents=latent_model_input,
                    timestep=t,
                    embeddings=encoded_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block,
                    mid_block_additional_residual=mid_block,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                denoised_latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents_for_view,
                    **extra_step_kwargs,
                ).prev_sample

                # Blur edges as needed
                multiplier = torch.ones_like(denoised_latents)
                if self.chunking_blur > 0:
                    chunking_blur_latent_size = self.chunking_blur // self.vae_scale_factor
                    blur_left = left > 0
                    blur_top = top > 0
                    blur_right = right < latent_width
                    blur_bottom = bottom < latent_height

                    for j in range(chunking_blur_latent_size):
                        mult = (j + 1) / (chunking_blur_latent_size + 1)
                        if blur_left:
                            multiplier[:, :, :, j] *= mult
                        if blur_top:
                            multiplier[:, :, j, :] *= mult
                        if blur_right:
                            multiplier[:, :, :, engine_latent_size - j - 1] *= mult
                        if blur_bottom:
                            multiplier[:, :, engine_latent_size - j - 1, :] *= mult

                value[:, :, top:bottom, left:right] += denoised_latents * multiplier
                count[:, :, top:bottom, left:right] += multiplier

                # Call the progress callback
                if progress_callback is not None:
                    progress_callback()

            # multidiffusion
            latents = torch.where(count > 0, value / count, value)

            # call the latent_callback, if provided
            if (
                latent_callback is not None
                and latent_callback_steps is not None
                and i % latent_callback_steps == 0
                and (
                    i == num_steps - 1
                    or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)
                )
            ):
                latent_callback_value = latents

                if latent_callback_type != "latent":
                    latent_callback_value = self.decode_latents(
                        latent_callback_value, device=device, progress_callback=progress_callback
                    )
                    latent_callback_value = self.denormalize_latents(latent_callback_value)
                    if latent_callback_type != "pt":
                        latent_callback_value = self.image_processor.pt_to_numpy(
                            latent_callback_value
                        )
                        if latent_callback_type == "pil":
                            latent_callback_value = self.image_processor.numpy_to_pil(
                                latent_callback_value
                            )
                latent_callback(latent_callback_value)

        return latents

    @torch.no_grad()
    def decode_latent_view(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Issues the command to decode a chunk of latents with the VAE.
        """
        return self.vae.decode(latents, return_dict=False)[0]

    @torch.no_grad()
    def decode_latents_unchunked(
        self, latents: torch.Tensor, device: Union[str, torch.device]
    ) -> torch.Tensor:
        """
        Decodes the latents using the VAE without chunking.
        """
        return self.decode_latent_view(latents).to(device=device)

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        device: Union[str, torch.device],
        progress_callback: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Decodes the latents in chunks as necessary.
        """
        samples, _, height, width = latents.shape
        height *= self.vae_scale_factor
        width *= self.vae_scale_factor

        latents = 1 / self.vae.config.scaling_factor * latents

        chunks = self.get_chunks(height, width)
        total_steps = len(chunks)

        if total_steps == 1:
            result = self.decode_latents_unchunked(latents, device)
            if progress_callback is not None:
                progress_callback()
            return result

        latent_width = width // self.vae_scale_factor
        latent_height = height // self.vae_scale_factor

        count = torch.zeros((samples, 3, height, width)).to(device=device)
        value = torch.zeros_like(count)

        logger.debug(f"Decoding latents in {total_steps} steps")

        # iterate over chunks
        for i, (top, bottom, left, right) in enumerate(chunks):
            # Slice latents
            latents_for_view = latents[:, :, top:bottom, left:right]

            # Get pixel indices
            top_px = top * self.vae_scale_factor
            bottom_px = bottom * self.vae_scale_factor
            left_px = left * self.vae_scale_factor
            right_px = right * self.vae_scale_factor

            # Decode latents
            decoded_latents = self.decode_latent_view(latents_for_view).to(device=device)

            # Blur edges as needed
            multiplier = torch.ones_like(decoded_latents)
            if self.chunking_blur > 0:
                chunking_blur_size = self.chunking_blur
                blur_left = left > 0
                blur_top = top > 0
                blur_right = right < latent_width
                blur_bottom = bottom < latent_height

                for j in range(chunking_blur_size):
                    mult = (j + 1) / (chunking_blur_size + 1)
                    if blur_left:
                        multiplier[:, :, :, j] *= mult
                    if blur_top:
                        multiplier[:, :, j, :] *= mult
                    if blur_right:
                        multiplier[:, :, :, self.engine_size - j - 1] *= mult
                    if blur_bottom:
                        multiplier[:, :, self.engine_size - j - 1, :] *= mult

            value[:, :, top_px:bottom_px, left_px:right_px] += decoded_latents * multiplier
            count[:, :, top_px:bottom_px, left_px:right_px] += multiplier

            if progress_callback is not None:
                progress_callback()

        # re-average pixels
        latents = torch.where(count > 0, value / count, value)
        return latents

    @torch.no_grad()
    def get_step_complete_callback(
        self,
        overall_steps: int,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        log_interval: int = 10,
        log_sampling_duration: Union[int, float] = 5,
    ) -> Callable[[], None]:
        """
        Creates a scoped callback to trigger during iterations
        """
        overall_step, window_start_step, its = 0, 0, 0.0
        window_start = datetime.datetime.now()
        digits = math.ceil(math.log10(overall_steps))

        def step_complete() -> None:
            nonlocal overall_step, window_start, window_start_step, its
            overall_step += 1
            if overall_step % log_interval == 0 or overall_step == overall_steps:
                seconds_in_window = (datetime.datetime.now() - window_start).total_seconds()
                its = (overall_step - window_start_step) / seconds_in_window
                unit = "s/it" if its < 1 else "it/s"
                its = 1 / its if its < 1 else its
                logger.debug(
                    f"{{0:0{digits}d}}/{{1:0{digits}d}}: {{2:0.2f}} {{3:s}}".format(
                        overall_step, overall_steps, its, unit
                    )
                )

                if seconds_in_window > log_sampling_duration:
                    window_start_step = overall_step
                    window_start = datetime.datetime.now()

            if progress_callback is not None:
                progress_callback(overall_step, overall_steps, its)

        return step_complete

    @torch.no_grad()
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
        """
        Invokes the pipeline.
        """
        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Allow overridding 'chunk'
        if chunking_size is not None:
            self.chunking_size = chunking_size
        if chunking_blur is not None:
            self.chunking_blur = chunking_blur

        # Define outputs here to process later
        prepared_latents: Optional[torch.Tensor] = None
        prepared_control_image: Optional[torch.Tensor] = None
        output_nsfw: Optional[List[bool]] = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("Prompt or prompt embeds are required.")

        if self.unet.config.in_channels == 9:
            if not image:
                logger.warning("No image present, but using inpainting model. Adding blank image.")
                image = PIL.Image.new("RGB", (width, height))
            if not mask:
                logger.warning("No mask present, but using inpainting model. Adding blank mask.")
                mask = PIL.Image.new("RGB", (width, height), (255, 255, 255))

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if image and not mask:
            timesteps, num_inference_steps = self.get_timesteps(
                num_inference_steps, strength, device
            )
        else:
            timesteps = self.scheduler.timesteps

        batch_size *= num_images_per_prompt

        # Calculate chunks
        num_chunks = len(self.get_chunks(height, width))

        # Calculate total steps
        chunked_steps = 1

        if image is not None:
            chunked_steps += 1
        if mask is not None:
            chunked_steps += 1
        if latent_callback is not None and latent_callback_steps is not None:
            chunked_steps += num_inference_steps // latent_callback_steps
        overall_num_steps = num_chunks * (num_inference_steps + chunked_steps)

        step_complete = self.get_step_complete_callback(overall_num_steps, progress_callback)

        with self.get_runtime_context(batch_size, device):
            # Base runtime has no context, but extensions do
            encoded_prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

            if image and isinstance(image, str):
                image = PIL.Image.open(image)

            if mask and isinstance(mask, str):
                mask = PIL.Image.open(mask)

            if scale_image and image:
                image_width, image_height = image.size
                if image_width != width or image_height != height:
                    logger.debug(
                        f"Resizing input image from {image_width}x{image_height} to {width}x{height}"
                    )
                    image = image.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])

            if scale_image and mask:
                mask_width, mask_height = mask.size
                if mask_width != width or mask_height != height:
                    logger.debug(
                        f"Resizing input mask from {mask_width}x{mask_height} to {width}x{height}"
                    )
                    mask = mask.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])

            if image:
                image = image.convert("RGB")
            if mask:
                mask = mask.convert("L")

            if image and mask:
                prepared_mask, prepared_image = self.prepare_mask_and_image(mask, image)
            elif image:
                prepared_image = self.image_processor.preprocess(image)
                prepared_mask, prepared_image_latents = None, None
            else:
                prepared_image, prepared_mask, prepared_image_latents = None, None, None

            if width < self.engine_size or height < self.engine_size:
                # Disable chunking
                logger.debug(f"{width}x{height} is smaller than is chunkable, disabling.")
                self.chunking_size = 0

            if prepared_image is not None and prepared_mask is not None:
                # Running the pipeline on an image with a mask
                num_channels_latents = self.vae.config.latent_channels

                if latents:
                    prepared_latents = latents.to(device) * self.schedule.init_noise_sigma
                else:
                    prepared_latents = self.create_latents(
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        encoded_prompt_embeds.dtype,
                        device,
                        generator,
                    )

                prepared_mask, prepared_image_latents = self.prepare_mask_latents(
                    prepared_mask,
                    prepared_image,
                    batch_size,
                    height,
                    width,
                    encoded_prompt_embeds.dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    progress_callback=step_complete,
                )

                num_channels_mask = prepared_mask.shape[1]
                num_channels_masked_image = prepared_image_latents.shape[1]

                if (
                    num_channels_latents + num_channels_mask + num_channels_masked_image
                    != self.unet.config.in_channels
                ):
                    raise ValueError(
                        f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                        f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                        f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                        f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                        " `pipeline.unet` or your `mask_image` or `image` input."
                    )
            elif prepared_image is not None:
                # Running the pipeline on an image, start with that
                prepared_latents = self.prepare_image_latents(
                    prepared_image,
                    timesteps[:1].repeat(batch_size),
                    batch_size,
                    encoded_prompt_embeds.dtype,
                    device,
                    generator,
                    progress_callback=step_complete,
                )
            elif latents:
                # Running the pipeline on existing latents, add some noise
                prepared_latents = latents.to(device) * self.self.scheduler.init_noise_sigma
            else:
                # Create random latents for the whole unet
                prepared_latents = self.create_latents(
                    batch_size,
                    self.unet.config.in_channels,
                    height,
                    width,
                    encoded_prompt_embeds.dtype,
                    device,
                    generator,
                )

            # Look for controlnet and conditioning image
            if control_image is not None:
                if self.controlnet is None:
                    logger.warning("Control image passed, but no controlnet present. Ignoring.")
                    prepared_control_image = None
                else:
                    if type(control_image) is str:
                        control_image = PIL.Image.open(control_image)
                    self.controlnet = self.controlnet.to(device=device)  # Cast to device
                    prepared_control_image = self.prepare_control_image(
                        image=control_image,
                        height=height,
                        width=width,
                        batch_size=batch_size,
                        num_images_per_prompt=num_images_per_prompt,
                        device=device,
                        dtype=self.controlnet.dtype,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                    )
            elif self.controlnet is not None:
                if image and mask:
                    logger.info(
                        "Assuming controlnet inpaint, creating conditioning image from image and mask"
                    )
                    prepared_control_image = self.prepare_controlnet_inpaint_control_image(
                        image=image, mask=image, device=device, dtype=self.controlnet.dtype
                    )
                else:
                    self.controlnet = self.controlnet.to("cpu")
                    logger.warning(
                        "Controlnet present, but no conditioning image. Disabling controlnet."
                    )

            # Should no longer be None
            prepared_latents = cast(torch.Tensor, prepared_latents)

            # 6. Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Make sure controlnet on device
            if self.controlnet is not None:
                self.controlnet = self.controlnet.to(device=device)

            # 7. Denoising loop
            prepared_latents = self.denoise(
                height=height,
                width=width,
                device=device,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                latents=prepared_latents,
                encoded_prompt_embeds=encoded_prompt_embeds,
                conditioning_scale=conditioning_scale,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mask=prepared_mask,
                mask_image=prepared_image_latents,
                control_image=prepared_control_image,
                progress_callback=step_complete,
                latent_callback=latent_callback,
                latent_callback_steps=latent_callback_steps,
                latent_callback_type=latent_callback_type,
                extra_step_kwargs=extra_step_kwargs,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            # Clear no longer needed tensors
            del prepared_mask
            del prepared_image_latents
            del prepared_control_image

            if output_type != "latent":
                prepared_latents = self.decode_latents(
                    prepared_latents, device=device, progress_callback=step_complete
                )

        if output_type == "latent":
            output = prepared_latents
        else:
            output = self.denormalize_latents(prepared_latents)
            if output_type != "pt":
                output = self.image_processor.pt_to_numpy(output)
                output_nsfw = self.run_safety_checker(output, device, encoded_prompt_embeds.dtype)[
                    1
                ]
                if output_type == "pil":
                    output = self.image_processor.numpy_to_pil(output)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if self.controlnet is not None:
            self.controlnet = self.controlnet.to("cpu")  # Unload controlnet from GPU

        if not return_dict:
            return (output, output_nsfw)

        return StableDiffusionPipelineOutput(images=output, nsfw_content_detected=output_nsfw)
