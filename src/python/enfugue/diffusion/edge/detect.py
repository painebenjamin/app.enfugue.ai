from __future__ import annotations

import cv2
import PIL
import numpy as np

from enfugue.util import check_download_to_dir
from enfugue.diffusion.vision import ComputerVision

__all__ = ["EdgeDetector"]


class EdgeDetector:
    """
    Provides edge detection methods
    """

    HED_PROTOTXT = "https://github.com/ashukid/hed-edge-detector/raw/master/deploy.prototxt"
    HED_CAFFEMODEL = (
        "https://github.com/ashukid/hed-edge-detector/raw/master/hed_pretrained_bsds.caffemodel"
    )
    HED_MEAN = (104.00698793, 116.66876762, 122.67891434)
    MLSD_WEIGHTS = "https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_large_512_fp32.pth"

    def __init__(self, model_dir: str) -> None:
        """
        On initialization, pass the model dir.
        """
        self.model_dir = model_dir

    @property
    def hed_prototxt(self) -> str:
        """
        Gets the local path to the HED prototxt.
        """
        return check_download_to_dir(self.HED_PROTOTXT, self.model_dir)

    @property
    def hed_caffemodel(self) -> str:
        """
        Gets the local path to the HED caffemodel.
        """
        return check_download_to_dir(self.HED_CAFFEMODEL, self.model_dir)

    @property
    def mlsd_weights(self) -> str:
        """
        Gets the local path to the MLSD weights file.
        """
        return check_download_to_dir(self.MLSD_WEIGHTS, self.model_dir)

    @staticmethod
    def canny(image: PIL.Image.Image, lower: int = 100, upper: int = 200) -> PIL.Image.Image:
        """
        Runs canny edge detection on an image.
        """
        canny = cv2.Canny(np.array(image), lower, upper)[:, :, None]
        return PIL.Image.fromarray(np.concatenate([canny, canny, canny], axis=2))

    def hed(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Runs holistically-nested edge detection on an image.
        """
        from enfugue.diffusion.edge.hed import HEDCropLayer  # type: ignore[attr-defined]

        width, height = image.size

        model = cv2.dnn.readNetFromCaffe(self.hed_prototxt, self.hed_caffemodel)
        cv2.dnn_registerLayer("Crop", HEDCropLayer)

        cv2_image = ComputerVision.convert_image(image)
        dnn_input = cv2.dnn.blobFromImage(
            cv2_image,
            scalefactor=1.0,
            size=(width, height),
            mean=self.HED_MEAN,
            swapRB=False,
            crop=False,
        )
        model.setInput(dnn_input)
        dnn_output = model.forward()
        output_array = (cv2.cvtColor(dnn_output[0, 0], cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
        cv2.dnn_unregisterLayer("Crop")
        return PIL.Image.fromarray(output_array)

    def mlsd(self, image: PIL.Image.Image) -> PIL.Image.Image:
        """
        Runs Mobile Line Segment Detection (MLSD) on an image.
        """
        import torch
        from enfugue.diffusion.edge.mlsd import MLSD, pred_lines  # type: ignore[attr-defined]

        if torch.cuda.is_available():
            model = MLSD().cuda().eval()
            device = torch.device("cuda")
        else:
            model = MLSD().eval()
            device = torch.device("cpu")

        model.load_state_dict(torch.load(self.mlsd_weights, map_location=device), strict=True)
        cv2_image = ComputerVision.convert_image(image)
        cv2_image = cv2.resize(cv2_image, (512, 512))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        lines = pred_lines(cv2_image, model, [512, 512], 0.1, 20)
        cv2_image = np.zeros((512, 512, 3), np.uint8)

        for line in lines:
            cv2.line(
                cv2_image,
                (int(line[0]), int(line[1])),
                (int(line[2]), int(line[3])),
                (255, 255, 255),
                1,
                16,
            )
        # Unload model and wipe cache
        del model
        torch.cuda.empty_cache()
        return ComputerVision.revert_image(cv2_image).resize(image.size)
