# type: ignore
# adapted from https://github.com/ProjectNUWA/DragNUWA

from enfugue.diffusion.animate.dragnuwa.svd.modules.autoencoding.losses.discriminator_loss import GeneralLPIPSWithDiscriminator
from enfugue.diffusion.animate.dragnuwa.svd.modules.autoencoding.losses.lpips import LatentLPIPS

GeneralLPIPSWithDiscriminator, LatentLPIPS # Silence importchecker

__all__ = [
    "GeneralLPIPSWithDiscriminator",
    "LatentLPIPS",
]
