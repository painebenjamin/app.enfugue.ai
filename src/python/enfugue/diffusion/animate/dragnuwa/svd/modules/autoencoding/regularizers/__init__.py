# type: ignore
from __future__ import annotations
# adapted from https://github.com/ProjectNUWA/DragNUWA
from typing import Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from enfugue.diffusion.animate.dragnuwa.svd.modules.distributions.distributions import DiagonalGaussianDistribution
from enfugue.diffusion.animate.dragnuwa.svd.modules.autoencoding.regularizers.base import AbstractRegularizer

class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: Tensor) -> Tuple[Tensor, dict]:
        import torch
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log
