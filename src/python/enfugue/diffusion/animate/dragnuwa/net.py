# type: ignore
# adapted from https://github.com/ProjectNUWA/DragNUWA
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enfugue.diffusion.animate.dragnuwa.utils import *

#### SVD
from enfugue.util import logger
from enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.video_model_flow import VideoUNet_flow
from enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.denoiser import Denoiser
from enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules import *
from enfugue.diffusion.animate.dragnuwa.svd.models.autoencoder import AutoencodingEngine
from enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.wrappers import OpenAIWrapper
from enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.sampling import EulerEDMSampler
from enfugue.diffusion.animate.dragnuwa.lora import (
    inject_trainable_lora,
    inject_trainable_lora_extended,
    extract_lora_ups_down,
)

def inject_lora(use_lora, model, replace_modules, is_extended=False, dropout=0.0, r=16):
    injector = (
        inject_trainable_lora if not is_extended else inject_trainable_lora_extended
    )

    params = None
    negation = None

    if use_lora:
        REPLACE_MODULES = replace_modules
        injector_args = {
            "model": model,
            "target_replace_module": REPLACE_MODULES,
            "r": r,
        }
        if not is_extended:
            injector_args["dropout_p"] = dropout

        params, negation = injector(**injector_args)
        for _up, _down in extract_lora_ups_down(
            model, target_replace_module=REPLACE_MODULES
        ):
            break

    return params, negation


@dataclass
class DragNUWANetArgs:
    """
    A data class holding the arguments for the network
    """

    fps: int = 4
    height: int = 320
    width: int = 576
    # lora
    unet_lora_rank: int = 32
    # model
    denoiser_config: Dict[str, Any] = field(default_factory=dict)
    network_config: Dict[str, Any] = field(default_factory=dict)
    first_stage_config: Dict[str, Any] = field(default_factory=dict)
    sampler_config: Dict[str, Any] = field(default_factory=dict)
    conditioner_emb_models: List[Dict[str, Any]] = field(default_factory=list)
    # VAE
    scale_factor: float = 0.18215
    # SVD
    num_frames: int = 14
    ### others
    seed: int = 42

    def __post_init__(self) -> None:
        self.denoiser_config = {
            **{
                "scaling_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.denoiser_scaling.VScalingWithEDMcNoise",
                }
            },
            **self.denoiser_config
        }
        self.network_config = {
            **{
                "adm_in_channels": 768,
                "num_classes": "sequential",
                "use_checkpoint": True,
                "in_channels": 8,
                "out_channels": 4,
                "model_channels": 320,
                "attention_resolutions": [4, 2, 1],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 4],
                "num_head_channels": 64,
                "use_linear_in_transformer": True,
                "transformer_depth": 1,
                "context_dim": 1024,
                "spatial_transformer_attn_type": "softmax-xformers",
                "extra_ff_mix_layer": True,
                "use_spatial_context": True,
                "merge_strategy": "learned_with_images",
                "video_kernel_size": [3, 1, 1],
                "flow_dim_scale": 1,
            },
            **self.network_config
        }
        self.first_stage_config = {
            **{
                "loss_config": {"target": "torch.nn.Identity"},
                "regularizer_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.autoencoding.regularizers.DiagonalGaussianRegularizer"
                },
                "encoder_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.model.Encoder",
                    "params": {
                        "attn_type": "vanilla",
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                },
                "decoder_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.autoencoding.temporal_ae.VideoDecoder",
                    "params": {
                        "attn_type": "vanilla",
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [1, 2, 4, 4],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                        "video_kernel_size": [3, 1, 1],
                    },
                },
            },
            **self.first_stage_config
        }
        self.sampler_config = {
            **{
                "discretization_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.discretizer.EDMDiscretization",
                    "params": {
                        "sigma_max": 700.0,
                    },
                },
                "guider_config": {
                    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.diffusionmodules.guiders.LinearPredictionGuider",
                    "params": {"max_scale": 2.5, "min_scale": 1.0, "num_frames": 14},
                },
                "num_steps": 25,
            },
            **self.sampler_config
        }

        input_keys = [model["input_key"] for model in self.conditioner_emb_models]
        for default_conditioner in [
            {
                "is_trainable": False,
                "input_key": "cond_frames_without_noise",  # crossattn
                "ucg_rate": 0.1,
                "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.FrozenOpenCLIPImagePredictionEmbedder",
                "params": {
                    "n_cond_frames": 1,
                    "n_copies": 1,
                    "open_clip_embedding_config": {
                        "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.FrozenOpenCLIPImageEmbedder",
                        "params": {
                            "freeze": True,
                        },
                    },
                },
            },
            {
                "input_key": "fps_id",  # vector
                "is_trainable": False,
                "ucg_rate": 0.1,
                "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND",
                "params": {
                    "outdim": 256,
                },
            },
            {
                "input_key": "motion_bucket_id",  # vector
                "ucg_rate": 0.1,
                "is_trainable": False,
                "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND",
                "params": {
                    "outdim": 256,
                },
            },
            {
                "input_key": "cond_frames",  # concat
                "is_trainable": False,
                "ucg_rate": 0.1,
                "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.VideoPredictionEmbedderWithEncoder",
                "params": {
                    "en_and_decode_n_samples_a_time": 1,
                    "disable_encoder_autocast": True,
                    "n_cond_frames": 1,
                    "n_copies": 1,
                    "is_ae": True,
                    "encoder_config": {
                        "target": "enfugue.diffusion.animate.dragnuwa.svd.models.autoencoder.AutoencoderKLModeOnly",
                        "params": {
                            "embed_dim": 4,
                            "monitor": "val/rec_loss",
                            "ddconfig": {
                                "attn_type": "vanilla-xformers",
                                "double_z": True,
                                "z_channels": 4,
                                "resolution": 256,
                                "in_channels": 3,
                                "out_ch": 3,
                                "ch": 128,
                                "ch_mult": [1, 2, 4, 4],
                                "num_res_blocks": 2,
                                "attn_resolutions": [],
                                "dropout": 0.0,
                            },
                            "lossconfig": {
                                "target": "torch.nn.Identity",
                            },
                        },
                    },
                },
            },
            {
                "input_key": "cond_aug",  # vector
                "ucg_rate": 0.1,
                "is_trainable": False,
                "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules.ConcatTimestepEmbedderND",
                "params": {
                    "outdim": 256,
                },
            },
        ]:
            if default_conditioner["input_key"] not in input_keys:
                self.conditioner_emb_models.append(default_conditioner)


    @classmethod
    def assemble(cls, **kwargs):
        import inspect

        signature = inspect.signature(cls).parameters
        from enfugue.util import logger

        net_kwargs = dict([(k, v) for k, v in kwargs.items() if k in signature])
        ignored_kwargs = set(list(kwargs.keys())) - set(list(net_kwargs.keys()))
        if ignored_kwargs:
            logger.warning(f"Keyword arguments ignored: {ignored_kwargs}")
        return cls(**net_kwargs)


def quick_freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


class DragNUWANet(nn.Module):
    def __init__(self, device="cpu", **kwargs):
        super(DragNUWANet, self).__init__()
        self.args = DragNUWANetArgs.assemble(**kwargs)
        self.device = device

        ### unet
        model = VideoUNet_flow(**self.args.network_config).to(self.device)
        self.model = OpenAIWrapper(model)

        ### denoiser and sampler
        self.denoiser = Denoiser(**self.args.denoiser_config)
        self.sampler = EulerEDMSampler(**self.args.sampler_config)

        ### conditioner
        self.conditioner = GeneralConditioner(self.args.conditioner_emb_models).to(self.device)

        ### first stage model
        self.first_stage_model = AutoencodingEngine(
            **self.args.first_stage_config
        ).eval().to(self.device)

        self.scale_factor = self.args.scale_factor
        self.en_and_decode_n_samples_a_time = (
            1  # decode 1 frame each time to save GPU memory
        )
        self.num_frames = self.args.num_frames
        unet_lora_params, unet_negation = inject_lora(
            True, self, ["OpenAIWrapper"], is_extended=False, r=self.args.unet_lora_rank
        )
        seed = (
            self.args.seed
            if self.args.seed is not None
            else random.randint(0, 2 ** 32 - 1)
        )
        self.seed = seed

    @property
    def seed(self) -> int:
        return self.args.seed

    @seed.setter
    def seed(self, new_seed: int) -> None:
        self.args.seed = new_seed
        os.environ["PL_GLOBAL_SEED"] = str(new_seed)
        random.seed(new_seed)
        np.random.seed(new_seed)
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed_all(new_seed)

    def to(self, *args, **kwargs):
        model_converted = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        self.sampler.device = self.device
        for embedder in self.conditioner.embedders:
            if hasattr(embedder, "device"):
                embedder.device = self.device
        return model_converted

    def train(self, *args):
        super().train(*args)
        self.conditioner.eval()
        self.first_stage_model.eval()

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = self.en_and_decode_n_samples_a_time  # 1
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        for n in range(n_rounds):
            kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
            out = self.first_stage_model.decode(
                z[n * n_samples : (n + 1) * n_samples], **kwargs
            )
            all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out
