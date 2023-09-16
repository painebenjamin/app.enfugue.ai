import cv2
import numpy as np

from PIL import Image

__all__ = ["ComputerVision"]

class ComputerVision:
    """
    Provides helper methods for cv2
    """
    @classmethod
    def show(cls, name: str, image: Image.Image) -> None:
        """
        Shows an image.
        Tries to use the Colab monkeypatch first, in case this is being ran in Colab.
        """
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(ComputerVision.convert_image(image))
        except:
            cv2.imshow(name, ComputerVision.convert_image(image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @classmethod
    def convert_image(cls, image: Image.Image) -> np.ndarray:
        """
        Converts PIL image to OpenCV format.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @classmethod
    def revert_image(cls, array: np.ndarray) -> Image.Image:
        """
        Converts OpenCV format to PIL image
        """
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
