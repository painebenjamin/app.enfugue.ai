from __future__ import annotations

from typing import Union, Literal, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.ndarray as NDArray
    from PIL.Image import Image
    

__all__ = ["ComputerVision"]

class ComputerVision:
    """
    Provides helper methods for cv2
    """
    @classmethod
    def show(cls, name: str, image: Image) -> None:
        """
        Shows an image.
        Tries to use the Colab monkeypatch first, in case this is being ran in Colab.
        """
        try:
            from google.colab.patches import cv2_imshow
            cv2_imshow(ComputerVision.convert_image(image))
        except:
            import cv2
            cv2.imshow(name, cls.convert_image(image))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @classmethod
    def convert_image(cls, image: Image) -> NDArray:
        """
        Converts PIL image to OpenCV format.
        """
        import cv2
        import numpy as np
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @classmethod
    def revert_image(cls, array: NDArray) -> Image:
        """
        Converts OpenCV format to PIL image
        """
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    @classmethod
    def tracking_features(
        cls,
        image: NDArray,
        max_corners: int=100,
        quality_level: float=0.3,
        min_distance: int=7,
        block_size: int=7,
    ) -> NDArray:
        """
        Gets good features to track from an image
        """
        import cv2
        from PIL import Image
        if isinstance(image, Image.Image):
            image = cls.convert_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.goodFeaturesToTrack(
            image,
            mask=None,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )


    @classmethod
    def sparse_flow(
        cls,
        image_1: Union[NDArray, Image],
        image_2: Union[NDArray, Image],
        features: Optional[List[Tuple[int, int]]]=None,
        feature_max_corners: int=100,
        feature_quality_level: float=0.3,
        feature_min_distance: int=7,
        feature_block_size: int=7,
        lk_window_size: Tuple[int, int]=(15, 15),
        lk_max_level: int=2,
        lk_criteria: Tuple[int, int, float] = (3, 10, 0.03),
    ) -> Tuple[NDArray, NDArray]:
        """
        Calculates the sparse optical flow of this video
        Adapted from https://github.com/spmallick/learnopencv/blob/master/Optical-Flow-in-OpenCV/algorithms/lucas_kanade.py
        """
        import cv2
        import numpy as np
        from math import floor
        from PIL import Image
        if isinstance(image_1, Image.Image):
            image_1 = cls.convert_image(image_1)
        if isinstance(image_2, Image.Image):
            image_2 = cls.convert_image(image_2)
        
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        # If no features passed, get them
        if features is None:
            features = cv2.goodFeaturesToTrack(
                image_1,
                mask=None,
                maxCorners=feature_max_corners,
                qualityLevel=feature_quality_level,
                minDistance=feature_min_distance,
                blockSize=feature_block_size
            )
        features_1, st, err = cv2.calcOpticalFlowPyrLK( # type: ignore[call-overload]
            image_1,
            image_2,
            features,
            None,
            winSize=lk_window_size,
            maxLevel=lk_max_level,
            criteria=lk_criteria
        )
        print(image_1.shape)
        print(image_2.shape)
        selected_new = features_1[st == 1]
        selected_old = features[st == 1]
        h, w = image_1.shape
        mask = np.zeros((h, w, 2)).astype(np.float32) # type: ignore[attr-defined]
        for i, (new, old) in enumerate(zip(selected_new, selected_old)):
            x1, y1 = new.ravel()
            x2, y2 = old.ravel()
            if 0 <= y1 < h and 0 <= x1 < w:
                mask[floor(y1), floor(x1), 0] = x2 - x1
                mask[floor(y1), floor(x1), 1] = y2 - y1
        features = selected_new.reshape(-1, 1, 2)
        return mask, features

    @classmethod
    def dense_flow(
        cls,
        image_1: Union[NDArray, Image],
        image_2: Union[NDArray, Image],
        method: Literal["dense-lucas-kanade", "farneback", "rlof"] = "dense-lucas-kanade",
        farneback_params: List[Union[float, int]] = [0.5, 3, 15, 3, 5, 1.2, 0], # Params for farneback method
    ) -> NDArray:
        """
        Calculates the dense optical flow of this video
        Adapted from https://github.com/spmallick/learnopencv/blob/master/Optical-Flow-in-OpenCV/algorithms/dense_optical_flow.py
        """
        from PIL import Image
        import cv2
        import numpy as np

        if isinstance(image_1, Image.Image):
            image_1 = cls.convert_image(image_1)
        if isinstance(image_2, Image.Image):
            image_2 = cls.convert_image(image_2)

        if method == "dense-lucas-kanade":
            flow = cv2.optflow.calcOpticalFlowSparseToDense( # type: ignore[attr-defined]
                cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY),
                None,
            )
        elif method == "farneback":
            flow = cv2.calcOpticalFlowFarneback( # type: ignore[call-overload]
                cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY),
                None,
                *farneback_params
            )
        elif method == "rlof":
            flow = cv2.optflow.calcOpticalFlowDenseRLOF( # type: ignore[attr-defined]
                image_1,
                image_2,
                None
             )
        else:
            raise IOError(f"Unknown dense optical flow method '{method}'") # type: ignore[unreachable]

        return flow

    @classmethod
    def flow_to_image(cls, flow: NDArray) -> Image:
        """
        Turns optical flow into an image
        """
        import cv2
        import numpy as np
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3)).astype(np.uint8) # type: ignore[attr-defined]
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # type: ignore[call-overload]
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return ComputerVision.revert_image(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
