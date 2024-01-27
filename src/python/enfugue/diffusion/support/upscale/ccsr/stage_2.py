# type: ignore

def get_model(state_dict):
    from enfugue.diffusion.support.upscale.ccsr.ldm.util import instantiate_from_config
    from enfugue.diffusion.support.upscale.ccsr.utils.common import load_state_dict

    model = instantiate_from_config({
        "target": "enfugue.diffusion.support.upscale.ccsr.model.ccsr_stage2.ControlLDM",
        "params": {
            "linear_start": 0.00085,
            "linear_end": 0.0120,
            "num_timesteps_cond": 1,
            "log_every_t": 200,
            "timesteps": 1000,
            "t_max": 0.6667,
            "t_min": 0.3333,
            "first_stage_key": "jpg",
            "cond_stage_key": "txt",
            "control_key": "hint",
            "image_size": 64,
            "channels": 4,
            "cond_stage_trainable": False,
            "conditioning_key": "crossattn",
            "monitor": "val/loss_simple_ema",
            "scale_factor": 0.18215,
            "use_ema": False,
            "sd_locked": True,
            "only_mid_control": False,
            "learning_rate": 5e-6,
            "control_stage_config": {
                "target": "enfugue.diffusion.support.upscale.ccsr.model.ccsr_stage2.ControlNet",
                "params": {
                    "use_checkpoint": True,
                    "image_size": 32,
                    "in_channels": 4,
                    "hint_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [ 4, 2, 1 ],
                    "num_res_blocks": 2,
                    "channel_mult": [ 1, 2, 4, 4 ],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 1024,
                    "legacy": False,
                },
            },
            "unet_config": {
                "target": "enfugue.diffusion.support.upscale.ccsr.model.ccsr_stage2.ControlledUnetModel",
                "params": {
                    "use_checkpoint": True,
                    "image_size": 32,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [ 4, 2, 1 ],
                    "num_res_blocks": 2,
                    "channel_mult": [ 1, 2, 4, 4 ],
                    "num_head_channels": 64,
                    "use_spatial_transformer": True,
                    "use_linear_in_transformer": True,
                    "transformer_depth": 1,
                    "context_dim": 1024,
                    "legacy": False,
                },
            },
            "first_stage_config": {
                "target": "enfugue.diffusion.support.upscale.ccsr.ldm.models.autoencoder.AutoencoderKL",
                "params": {
                    "embed_dim": 4,
                    "monitor": "val/rec_loss",
                    "ddconfig": {
                        "double_z": True,
                        "z_channels": 4,
                        "resolution": 256,
                        "in_channels": 3,
                        "out_ch": 3,
                        "ch": 128,
                        "ch_mult": [ 1, 2, 4, 4 ],
                        "num_res_blocks": 2,
                        "attn_resolutions": [],
                        "dropout": 0.0,
                    },
                    "lossconfig": {
                        "target": "torch.nn.Identity",
                    },
                },
            },
            "cond_stage_config": {
                "target": "enfugue.diffusion.support.upscale.ccsr.ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder",
                "params": {
                    "freeze": True,
                    "layer": "penultimate",
                },
            },
            "lossconfig": {
                "target": "enfugue.diffusion.support.upscale.ccsr.ldm.modules.losses.LPIPSWithDiscriminator",
                "params": {
                    "disc_start": 1.0,
                    "kl_weight": 0,
                    "perceptual_weight": 1.0,
                    "disc_weight": 0.5,
                    "disc_factor": 1.0,
                },
            },
        },
    },)
    load_state_dict(model, state_dict)
    return model
