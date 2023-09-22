# Inspired by the following:
# https://github.com/ddowd97/Python-Image-Morpher/blob/master/Morphing/Morphing.py
# https://github.com/jankovicsandras/autoimagemorph/blob/master/autoimagemorph.py
# https://github.com/spmallick/learnopencv/blob/master/FaceMorph/faceMorph.py
import cv2
import numpy as np

from typing import Union, Literal, Iterator, Tuple, List, Any

from PIL import Image
from scipy.spatial import Delaunay
from matplotlib.path import Path

class Triangle:
    """
    Stores vertices for a triangle and allows some calculations.
    """
    def __init__(self, vertices: np.ndarray) -> None:
        self.vertices = vertices

    @property
    def points(self) -> np.ndarray:
        """
        Gets the points contained within this triangle.
        """
        if not hasattr(self, "_points"):
            self.min_x = int(self.vertices[:, 0].min())
            self.max_x = int(self.vertices[:, 0].max())
            self.min_y = int(self.vertices[:, 1].min())
            self.max_y = int(self.vertices[:, 1].max())
            x_list = range(self.min_x, self.max_x + 1)
            y_list = range(self.min_y, self.max_y + 1)
            point_list = [(x, y) for x in x_list for y in y_list]

            points = np.array(point_list, np.float64)
            p = Path(self.vertices)
            grid = p.contains_points(points)
            mask = grid.reshape(self.max_x - self.min_x + 1, self.max_y + self.min_y + 1)
            filtered = np.where(np.array(mask) == True)
            
            self._points = np.vstack((filtered[0] + self.min_x, filtered[1] + self.min_y, np.ones(filtered[0].shape[0])))
        return self._points


class Morpher:
    """
    A quick-calculating morpher class that allows you to morph between two images.
    """
    def __init__(
        self,
        left: Union[str, np.ndarray, Image.Image],
        right: Union[str, np.ndarray, Image.Image],
        features: int = 8
    ) -> None:
        from enfugue.diffusion.util import ComputerVision
        if isinstance(left, str):
            left = Image.open(left)
        if isinstance(left, np.ndarray):
            left = ComputerVision.revert_image(left)
        if isinstance(right, str):
            right = Image.open(right)
        if isinstance(right, np.ndarray):
            right = ComputerVision.revert_image(right)
        self.left = left
        self.right = right.resize(self.left.size)
        self.features = features

    @property
    def start(self) -> np.ndarray:
        """
        Gets the starting image in OpenCV format.
        """
        from enfugue.diffusion.util import ComputerVision
        if not hasattr(self, "_start"):
            self._start = ComputerVision.convert_image(self.left)
        return self._start

    @property
    def end(self) -> np.ndarray:
        """
        Gets the ending image in OpenCV format.
        """
        from enfugue.diffusion.util import ComputerVision
        if not hasattr(self, "_end"):
            self._end = ComputerVision.convert_image(self.right)
        return self._end
        
    @property
    def start_points(self) -> List[List[int]]:
        """
        Gets feature points from the left (start)
        """
        if not hasattr(self, "_start_points"):
            self._start_points = self.get_image_feature_points(self.start)
        return self._start_points

    @property
    def end_points(self) -> List[List[int]]:
        """
        Gets feature points from the left (end)
        """
        if not hasattr(self, "_end_points"):
            self._end_points = self.get_image_feature_points(self.end)
        return self._end_points

    @property
    def triangles(self) -> Iterator[Tuple[Triangle, Triangle]]:
        """
        Iterate over the tesselated triangles.
        """
        start = np.array(self.start_points, np.float64)
        end = np.array(self.end_points, np.float64)

        tesselated = Delaunay(start)

        start_np = start[tesselated.simplices]
        end_np = end[tesselated.simplices]

        for x, y in zip(start_np, end_np):
            yield (Triangle(x), Triangle(y))

    def get_image_feature_points(self, image: np.ndarray):
        """
        Gets tracked features for an image
        """
        height, width, channels = image.shape
        # Initialize points with four corners
        points = [
            [0, 0],
            [width - 1, 0],
            [0, height-1],
            [width-1, height-1]
        ]

        # Get height and width of cells
        h = int(height / self.features) - 1
        w = int(width / self.features) - 1

        # Iterate over cells
        for i in range(self.features):
            for j in range(self.features):
                # Crop and find feature point in frame
                cropped = image[(j*h):(j*h)+h, (i*w):(i*w)+w]
                monochrome = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                features = cv2.goodFeaturesToTrack(monochrome, 1, 0.1, 10) # Tunable
                if features is None:
                    # If there's nothing worth tracking in this cell, make our point the center
                    features = [[[h/2, w/2]]]

                # Go through features and add coordinates to array
                features = np.int0(features) # type: ignore[attr-defined]
                for feature in features:
                    x, y = feature.ravel()
                    x += (i*w)
                    y += (j*h)
                    points.append([x, y])

        # Return 4 + (features ^ 2) points
        return points

    def morph(
        self,
        start: np.ndarray,
        end: np.ndarray,
        target: np.ndarray,
        start_triangle: Triangle,
        end_triangle: Triangle,
        target_triangle: Triangle,
        alpha: float
    ) -> None:
        """
        Morphs into a target array at a certain alpha
        """
        # Find bounding boundangle for each triangle
        start_bound = cv2.boundingRect(np.float32([start_triangle.vertices])) # type: ignore[arg-type]
        end_bound = cv2.boundingRect(np.float32([end_triangle.vertices])) # type: ignore[arg-type]
        target_bound = cv2.boundingRect(np.float32([target_triangle.vertices])) # type: ignore[arg-type]

        # Offset points by left top corner of the respective rectangles
        start_rect = []
        end_rect = []
        target_rect = []

        for i in range(0, 3):
            target_rect.append((
                (target_triangle.vertices[i][0] - target_bound[0]),
                (target_triangle.vertices[i][1] - target_bound[1])
            ))
            start_rect.append((
                (start_triangle.vertices[i][0] - start_bound[0]),
                (start_triangle.vertices[i][1] - start_bound[1])
            ))
            end_rect.append((
                (end_triangle.vertices[i][0] - end_bound[0]),
                (end_triangle.vertices[i][1] - end_bound[1])
            ))

        # Get mask by filling triangle
        mask = np.zeros((target_bound[3], target_bound[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(target_rect), (1.0, 1.0, 1.0), 16, 0) # type: ignore[arg-type]

        # Apply warpImage to small rectangular patches
        start_image = start[
            start_bound[1]:start_bound[1] + start_bound[3],
            start_bound[0]:start_bound[0] + start_bound[2]
        ]
        end_image = end[
            end_bound[1]:end_bound[1] + end_bound[3],
            end_bound[0]:end_bound[0] + end_bound[2]
        ]
        size = (target_bound[2], target_bound[3])

        warp_start = self.affine(start_image, start_rect, target_rect, size)
        warp_end = self.affine(end_image, end_rect, target_rect, size)
        
        blend = (1.0 - alpha) * warp_start + alpha * warp_end

        # Copy triangular region of the rectangular patch to the output image
        target[
            target_bound[1]:target_bound[1]+target_bound[3],
            target_bound[0]:target_bound[0]+target_bound[2]
        ] = target[
            target_bound[1]:target_bound[1]+target_bound[3],
            target_bound[0]:target_bound[0]+target_bound[2]
        ] * (1 - mask) + blend * mask

    def affine(
        self,
        source: np.ndarray,
        source_rect: List[Tuple[Any, Any]],
        target_rect: List[Tuple[Any, Any]],
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Applies the affine transform for a section of an image
        """
        warp_mat = cv2.getAffineTransform(np.float32(source_rect), np.float32(target_rect)) # type: ignore[arg-type]
        return cv2.warpAffine(
            source,
            warp_mat,
            (size[0], size[1]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    def __call__(self, alpha: float, return_type: Literal["pil", "np"] = "pil"):
        """
        Gets an image at a particular point between 0 and 1
        """
        from enfugue.diffusion.util import ComputerVision
        start = np.float32(self.start)
        end = np.float32(self.end)
        target = np.zeros(start.shape, dtype=start.dtype)
        
        points = []
        for ((start_x, start_y), (end_x, end_y)) in zip(self.start_points, self.end_points):
            points.append([
                (1 - alpha) * start_x + alpha * end_x,
                (1 - alpha) * start_y + alpha * end_y,
            ])

        for start_tri, end_tri in self.triangles:
            target_tri = Triangle((1 - alpha) * start_tri.vertices + end_tri.vertices * alpha)
            self.morph(start, end, target, start_tri, end_tri, target_tri, alpha) # type: ignore[arg-type]

        target = np.uint8(target) # type: ignore[assignment]
        if return_type == "pil":
            return ComputerVision.revert_image(target)
        return target

    def save_video(
        self,
        path: str,
        length: int = 20,
        rate: float = 20.,
        overwrite: bool = False,
    ) -> int:
        """
        Saves the warped image(s) to an .mp4
        """
        from enfugue.diffusion.util import Video
        return Video([
            self(i/length)
            for i in range(length+1)
        ]).save(
            path,
            rate=rate,
            overwrite=overwrite
        )
