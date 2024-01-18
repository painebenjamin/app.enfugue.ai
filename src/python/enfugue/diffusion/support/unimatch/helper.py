from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

from enfugue.diffusion.support.model import SupportModel
from contextlib import contextmanager

if TYPE_CHECKING:
    import torch
    import numpy as np
    from PIL import Image
    from enfugue.diffusion.support.unimatch.unimatch import UniMatch # type: ignore[attr-defined]

__all__ = ["Unimatch"]

class UnimatchImageProcessor:
    """
    Used to process between two images.
    """
    def __init__(
        self,
        model: UniMatch,
        device: torch.device
    ) -> None:
        self.model = model
        self.device = device

    def __call__(
        self,
        image1: Image.Image,
        image2: Image.Image,
        padding_factor: int=32,
    ) -> np.ndarray:
        """
        Calculates optical flow between two images
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        image1 = np.array(image1).astype(np.uint8) # type: ignore[attr-defined]
        image2 = np.array(image2).astype(np.uint8) # type: ignore[attr-defined]

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

        # the model is trained with size: width > height
        transpose_img = False
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [
            int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor
        ]

        ori_size = image1.shape[-2:]

        # resize before inference
        if nearest_size[0] != ori_size[0] or nearest_size[1] != ori_size[1]:
            image1 = F.interpolate(
                image1,
                size=nearest_size,
                mode='bilinear',
                align_corners=True
            )
            image2 = F.interpolate(
                image2,
                size=nearest_size,
                mode='bilinear',
                align_corners=True
            )

        with torch.no_grad():
            results_dict = self.model(
                image1,
                image2,
                attn_type="swin",
                attn_splits_list=[2,8],
                corr_radius_list=[-1,4],
                prop_radius_list=[-1,1],
                num_reg_refine=6,
                task="flow",
                prep_bidir_flow=False
            )

            flow_pr = results_dict["flow_preds"][-1]  # [B, 2, H, W]

            # resize back
            if nearest_size[0] != ori_size[0] or nearest_size[1] != ori_size[1]:
                flow_pr = F.interpolate(
                    flow_pr,
                    size=ori_size,
                    mode="bilinear",
                    align_corners=True
                )
                flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / nearest_size[-1]
                flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / nearest_size[-2]

            if transpose_img:
                flow_pr = torch.transpose(flow_pr, -2, -1)

            flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
            return flow

class Unimatch(SupportModel):
    """
    Used to remove backgrounds from images automatically
    """
    FLOW_NET_PATH = "https://huggingface.co/spaces/haofeixu/unimatch/resolve/main/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"

    @contextmanager
    def flow(
        self,
        feature_channels: int=128,
        num_scales: int=2,
        upsample_factor: int=4,
        num_heads: int=1,
        ffn_dim_expansion: int=4,
        num_transformer_layers: int=6,
        regression_refinement: bool=True,
    ) -> Iterator[UnimatchImageProcessor]:
        """
        Instantiate the flow processor and return the callable
        """
        import torch
        from enfugue.diffusion.support.unimatch.unimatch import UniMatch # type: ignore
        from enfugue.diffusion.util import load_state_dict
        model_path = self.get_model_file(self.FLOW_NET_PATH)
        with self.context():
            model = UniMatch(
                feature_channels=feature_channels,
                 num_scales=num_scales,
                 upsample_factor=upsample_factor,
                 num_head=num_heads,
                 ffn_dim_expansion=ffn_dim_expansion,
                 num_transformer_layers=num_transformer_layers,
                 reg_refine=regression_refinement,
                 task="flow"
            )
            model.eval().to(device=self.device)
            state_dict = load_state_dict(model_path)
            model.load_state_dict(state_dict["model"], strict=False) # type: ignore[arg-type]
            del state_dict
            processor = UnimatchImageProcessor(model=model, device=self.device)
            yield processor
            del processor
            del model
