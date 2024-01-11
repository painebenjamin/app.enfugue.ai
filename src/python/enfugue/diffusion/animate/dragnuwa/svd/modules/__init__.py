# type: ignore
# adapted from https://github.com/ProjectNUWA/DragNUWA
from enfugue.diffusion.animate.dragnuwa.svd.modules.encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "enfugue.diffusion.animate.dragnuwa.svd.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
