import cv2
import numpy as np

from PIL import Image

from typing import Union, Literal

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

    @classmethod
    def noise(
        cls,
        image: Union[np.ndarray, Image.Image],
        method: Literal["gaussian", "poisson", "speckle", "salt-and-pepper"] = "poisson",
        gaussian_mean: Union[int, float] = 0.0,
        gaussian_variance: float = 0.01,
        poisson_factor: Union[int, float] = 2.25,
        salt_pepper_ratio: float = 0.5,
        salt_pepper_amount: float = 0.004,
        speckle_amount: float = 0.01,
    ) -> Union[np.ndarray, Image.Image]:
        """
        Adds noise to an image.
        """
        return_pil = isinstance(image, Image.Image)
        if return_pil:
            image = cls.convert_image(image)
        image = image.astype(np.float64) / 255.0
        width, height, channels = image.shape
        if method == "gaussian":
            gaussian_sigma = gaussian_variance ** 0.5
            gaussian = np.random.normal(
                gaussian_mean,
                gaussian_sigma,
                (width, height, channels)
            )
            gaussian = gaussian.reshape(width, height, channels)
            image += gaussian
        elif method == "salt-and-pepper":
            output = np.copy(image)
            # Do salt
            salt = np.ceil(salt_pepper_amount * image.size * salt_pepper_ratio)
            coordinates = [
                np.random.randint(0, i - 1, int(salt))
                for i in image.shape
            ]
            output[coordinates] = 1
            # Do pepper
            pepper = np.ceil(salt_pepper_amount * image.size * (1.0 - salt_pepper_ratio))
            coordinates = [
                np.random.randint(0, i - 1, int(pepper))
                for i in image.shape
            ]
            output[coordinates] = 0
            image = output
        elif method == "poisson":
            distinct_values = len(np.unique(image))
            distinct_values = poisson_factor ** np.ceil(np.log2(distinct_values))
            image = np.random.poisson(image * distinct_values) / float(distinct_values)
        elif method == "speckle":
            speckled = np.random.randn(width, height, channels)
            speckled = speckled.reshape(width, height, channels)
            image += (image * speckled * speckle_amount)
        else:
            raise ValueError(f"Unknown noise method {method}") # type: ignore[unreachable]
        image *= 255.0
        image = image.astype(np.uint8)
        if return_pil:
            return cls.revert_image(image)
        return image
