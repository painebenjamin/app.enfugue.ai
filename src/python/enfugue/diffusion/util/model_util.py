from __future__ import annotations

import os
import re

from enfugue.util import logger, check_download
from enfugue.diffusion.constants import *

from diffusers.utils.constants import DIFFUSERS_CACHE

from typing import Optional, Union, Literal, Dict, Tuple, Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        KarrasDiffusionSchedulers
    )

__all__ = [
    "get_scheduler",
    "get_controlnet",
    "get_vae",
    "ModelMerger"
]

def get_controlnet(
    controlnet: str,
    is_sdxl: bool = False,
    cache_dir: str = DIFFUSERS_CACHE,
    torch_dtype: Optional[torch.dtype] = None,
) -> ControlNetModel:
    """
    Gets a controlnet by name or path.
    """
    # Get paths by name if we aren't passed a repo
    is_file = ".safetensors" in controlnet or ".ckpt" in controlnet
    is_repo = "/" in controlnet and not is_file

    if not is_repo and not is_file:
        if is_sdxl:
            if controlnet == "canny":
                controlnet = CONTROLNET_CANNY_XL
            elif controlnet == "depth":
                controlnet = CONTROLNET_DEPTH_XL
            elif controlnet == "pose":
                controlnet = CONTROLNET_POSE_XL
            elif controlnet == "temporal":
                controlnet = CONTROLNET_TEMPORAL_XL
            else:
                raise ValueError(f"Unknown or unsupported ControlNet {controlnet}")
        else:
            if controlnet == "canny":
                controlnet = CONTROLNET_CANNY
            elif controlnet == "mlsd":
                controlnet = CONTROLNET_MLSD
            elif controlnet == "hed":
                controlnet = CONTROLNET_HED
            elif controlnet == "tile":
                controlnet = CONTROLNET_TILE
            elif controlnet == "scribble":
                controlnet = CONTROLNET_SCRIBBLE
            elif controlnet == "inpaint":
                controlnet = CONTROLNET_INPAINT
            elif controlnet == "depth":
                controlnet = CONTROLNET_DEPTH
            elif controlnet == "normal":
                controlnet = CONTROLNET_NORMAL
            elif controlnet == "pose":
                controlnet = CONTROLNET_POSE
            elif controlnet == "line":
                controlnet = CONTROLNET_LINE
            elif controlnet == "anime":
                controlnet = CONTROLNET_ANIME
            elif controlnet == "pidi":
                controlnet = CONTROLNET_PIDI
            elif controlnet == "temporal":
                controlnet = CONTROLNET_TEMPORAL
            else:
                raise ValueError(f"Unknown or unsupported ControlNet {controlnet}")
        is_repo = True
    from diffusers.models import ControlNetModel
    if is_file:
        if controlnet.startswith("http"):
            expected_controlnet_location = os.path.join(cache_dir, os.path.basename(controlnet))
            check_download(controlnet, expected_controlnet_location)
            controlnet = expected_controlnet_location

        return ControlNetModel.from_single_file(
            controlnet,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
    return ControlNetModel.from_pretrained(
        controlnet,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )

def get_vae(
    vae: str,
    cache_dir: str = DIFFUSERS_CACHE,
    torch_dtype: Optional[torch.dtype] = None,
) -> AutoencoderKL:
    # Get paths by name if we aren't passed a repo
    is_file = ".safetensors" in vae or ".ckpt" in vae
    is_repo = "/" in vae and not is_file
    if not is_repo and not is_file:
        if vae == "ema":
            vae = VAE_EMA
        elif vae == "mse":
            vae = VAE_MSE
        elif vae == "xl":
            vae = VAE_XL
        elif vae == "xl16":
            vae = VAE_XL16
        else:
            raise ValueError(f"Unknown or unsupported VAE {vae}")
    from diffusers.models import AutoencoderKL
    if is_file:
        if vae.startswith("http"):
            expected_vae_location = os.path.join(cache_dir, os.path.basename(vae))
            check_download(vae, expected_vae_location)
            vae = expected_vae_location

        return AutoencoderKL.from_single_file(
            vae,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
    return AutoencoderKL.from_pretrained(
        vae,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )

def get_scheduler(
    scheduler: SCHEDULER_LITERAL
) -> Tuple[KarrasDiffusionSchedulers, Dict[str, Any]]:
    """
    Gets a scheduler class with configuration
    """
    if scheduler == "ddim":
        from diffusers.schedulers import DDIMScheduler
        return DDIMScheduler, {}
    elif scheduler == "ddpm":
        from diffusers.schedulers import DDPMScheduler
        return DDPMScheduler, {}
    elif scheduler == "deis":
        from diffusers.schedulers import DEISMultistepScheduler
        return DEISMultistepScheduler, {}
    elif scheduler in ["dpmsm", "dpmsmk", "dpmsmka"]:
        from diffusers.schedulers import DPMSolverMultistepScheduler
        kwargs: Dict[str, Any] = {}
        if scheduler in ["dpmsmk", "dpmsmka"]:
            kwargs["use_karras_sigmas"] = True
            if scheduler == "dpmsmka":
                kwargs["algorithm_type"] = "sde-dpmsolver++"
        return DPMSolverMultistepScheduler, kwargs
    elif scheduler == "dpmss":
        from diffusers.schedulers import DPMSolverSinglestepScheduler
        return DPMSolverSinglestepScheduler, {}
    elif scheduler == "heun":
        from diffusers.schedulers import HeunDiscreteScheduler
        return HeunDiscreteScheduler, {}
    elif scheduler == "dpmd":
        from diffusers.schedulers import KDPM2DiscreteScheduler
        return KDPM2DiscreteScheduler, {}
    elif scheduler == "adpmd":
        from diffusers.schedulers import KDPM2AncestralDiscreteScheduler
        return KDPM2AncestralDiscreteScheduler, {}
    elif scheduler == "dpmsde":
        from diffusers.schedulers import DPMSolverSDEScheduler
        return DPMSolverSDEScheduler, {}
    elif scheduler == "unipc":
        from diffusers.schedulers import UniPCMultistepScheduler
        return UniPCMultistepScheduler, {}
    elif scheduler == "lmsd":
        from diffusers.schedulers import LMSDiscreteScheduler
        return LMSDiscreteScheduler, {}
    elif scheduler == "pndm":
        from diffusers.schedulers import PNDMScheduler
        return PNDMScheduler, {}
    elif scheduler == "eds":
        from diffusers.schedulers import EulerDiscreteScheduler
        return EulerDiscreteScheduler, {}
    elif scheduler == "eads":
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        return EulerAncestralDiscreteScheduler, {}
    raise ValueError(f"Unknown scheduler {scheduler}")

class ModelMerger:
    """
    Allows merging various Stable Diffusion models of various sizes.
    Inspired by https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/extras.py
    """

    CHECKPOINT_DICT_REPLACEMENTS = {
        "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
        "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
        "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
    }

    CHECKPOINT_DICT_SKIP = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

    discard_weights: Optional[re.Pattern]

    def __init__(
        self,
        primary_model: str,
        secondary_model: Optional[str],
        tertiary_model: Optional[str],
        interpolation: Optional[Literal["add-difference", "weighted-sum"]] = None,
        multiplier: Union[int, float] = 1.0,
        half: bool = True,
        discard_weights: Optional[Union[str, re.Pattern]] = None,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.tertiary_model = tertiary_model
        self.interpolation = interpolation
        self.multiplier = multiplier
        self.half = half
        if type(discard_weights) is str:
            self.discard_weights = re.compile(discard_weights)
        else:
            self.discard_weights = cast(Optional[re.Pattern], discard_weights)

    @staticmethod
    def as_half(tensor: torch.Tensor) -> torch.Tensor:
        """
        Halves a tensor if necessary
        """
        if tensor.dtype == torch.float:
            return tensor.half()
        return tensor

    @staticmethod
    def get_difference(theta0: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
        """
        Simply gets the difference from two values.
        """
        return theta0 - theta1

    @staticmethod
    def weighted_sum(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """
        Returns the sum of θ0 and θ1 weighted by ɑ
        """
        return ((1 - alpha) * theta0) + (alpha * theta1)

    @staticmethod
    def add_weighted_difference(theta0: torch.Tensor, theta1: torch.Tensor, alpha: Union[int, float]) -> torch.Tensor:
        """
        Adds a weighted difference back to the original value
        """
        return theta0 + (alpha * theta1)

    @staticmethod
    def get_state_dict_from_checkpoint(checkpoint: Dict) -> Dict:
        """
        Reads the raw state dictionary and find the proper state dictionary.
        """
        state_dict = checkpoint.pop("state_dict", checkpoint)
        if "state_dict" in state_dict:
            del state_dict["state_dict"]  # Remove any sub-embedded state dicts

        transformed_dict = dict(
            [(ModelMerger.transform_checkpoint_key(key), value) for key, value in state_dict.items()]
        )
        state_dict.clear()
        state_dict.update(transformed_dict)
        return state_dict

    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict:
        """
        Loads a checkpoint"s state dictionary on the CPU for model merging.
        """
        _, ext = os.path.splitext(checkpoint_path)
        logger.debug(f"Model merger loading {checkpoint_path}")
        if ext.lower() == ".safetensors":
            import safetensors.torch
            ckpt = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            import torch
            ckpt = torch.load(checkpoint_path, map_location="cpu")

        return ModelMerger.get_state_dict_from_checkpoint(ckpt)

    @staticmethod
    def is_ignored_key(key: str) -> bool:
        """
        Checks if a key should be ignored during merge.
        """
        return "model" not in key or key in ModelMerger.CHECKPOINT_DICT_SKIP

    @staticmethod
    def transform_checkpoint_key(text: str) -> str:
        """
        Transform a checkpoint key, if needed.
        """
        for key, value in ModelMerger.CHECKPOINT_DICT_REPLACEMENTS.items():
            if key.startswith(text):
                text = value + text[len(key) :]
        return text

    def save(self, output_path: str) -> None:
        """
        Runs the configured merger.
        """
        import torch
        logger.debug(
            f"Executing model merger with interpolation '{self.interpolation}', primary model {self.primary_model}, secondary model {self.secondary_model}, tertiary model {self.tertiary_model}"
        )

        primary_state_dict = self.load_checkpoint(self.primary_model)
        secondary_state_dict = None if not self.secondary_model else self.load_checkpoint(self.secondary_model)
        tertiary_state_dict = None if not self.tertiary_model else self.load_checkpoint(self.tertiary_model)

        theta_0 = primary_state_dict
        theta_1 = secondary_state_dict

        if self.interpolation == "add-difference":
            if theta_1 is None or tertiary_state_dict is None:
                raise ValueError(f"{self.interpolation} requires three models.")
            logger.debug("Merging secondary and tertiary models.")
            for key in theta_1.keys():
                if self.is_ignored_key(key):
                    continue
                if key in tertiary_state_dict:
                    theta_1[key] = self.get_difference(theta_1[key], tertiary_state_dict[key])
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])
            del tertiary_state_dict

        if self.interpolation == "add-difference":
            interpolate = self.add_weighted_difference
        else:
            interpolate = self.weighted_sum

        if theta_1 is not None:
            logger.debug("Merging primary and secondary models.")
            for key in theta_0.keys():
                if key not in theta_1 or self.is_ignored_key(key):
                    continue

                a = theta_0[key]
                b = theta_1[key]

                # Check if we're merging 4-channel (standard), 8-channel (ip2p) ir 9-channel (inpainting) model(s)
                if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                    if a.shape[1] == 4 and b.shape[1] == 9:
                        raise RuntimeError(
                            "When merging an inpainting model with a standard one, the primary model must be the inpainting model."
                        )
                    if a.shape[1] == 4 and b.shape[1] == 8:
                        raise RuntimeError(
                            "When merging an instruct-pix2pix model with a standard one, the primary model must be the instruct-pix2pix model."
                        )

                    if a.shape[1] == 8 and b.shape[1] == 4:
                        # Merging IP2P into Normal
                        theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, self.multiplier)
                        result_is_instruct_pix2pix_model = True
                    else:
                        # Merging inpainting into Normal
                        assert (
                            a.shape[1] == 9 and b.shape[1] == 4
                        ), f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"
                        theta_0[key][:, 0:4, :, :] = interpolate(a[:, 0:4, :, :], b, self.multiplier)
                        result_is_inpainting_model = True
                else:
                    theta_0[key] = interpolate(a, b, self.multiplier)

                if self.half:
                    theta_0[key] = self.as_half(theta_0[key])

            del theta_1

        if self.discard_weights is not None:
            for key in list(theta_0):
                if re.search(self.discard_weights, key):
                    theta_0.pop(key, None)

        logger.debug(f"Saving merged model to {output_path}")
        _, extension = os.path.splitext(output_path)
        if extension.lower() == ".safetensors":
            import safetensors.torch
            safetensors.torch.save_file(theta_0, output_path)
        else:
            torch.save(theta_0, output_path)
