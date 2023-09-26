from __future__ import annotations

from pibble.util.strings import Serializer
from typing import Any, Dict, List, Tuple, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from enfugue.diffusion.manager import DiffusionPipelineManager
    from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "GridMaker"
]

class GridMaker:
    """
    A small class for building grids.
    Usually this is to test how modifying parameters works.
    """
    def __init__(
        self,
        seed: int = 12345,
        grid_size: int = 256,
        grid_columns: int = 4,
        caption_height: int = 50,
        use_video: bool = False,
        **base_kwargs: Any
    ) -> None:
        self.seed = seed
        self.grid_size = grid_size
        self.grid_columns = grid_columns
        self.caption_height = caption_height
        self.base_kwargs = base_kwargs
        self.use_video = use_video

    @property
    def font(self) -> ImageFont:
        """
        Gets the default system font.
        """
        if not hasattr(self, "_font"):
            from PIL import ImageFont
            self._font = ImageFont.load_default()
        return self._font

    @property
    def text_max_length(self) -> int:
        """
        Calculates the maximum length of text
        """
        return 8 + self.grid_size // 8

    def split_text(self, text: str) -> str:
        """
        Splits text into lines based on grid size
        """
        max_length = self.text_max_length
        line_count = 1 + len(text) // max_length
        return "\n".join([
            text[(i*max_length):((i+1)*max_length)]
            for i in range(line_count)
        ])

    def format_parameter(self, parameter: Any) -> str:
        """
        Formats an individual parameter.
        """
        from torch import Tensor
        from PIL import Image
        if isinstance(parameter, Image.Image):
            width, height = parameter.size
            return f"Image({width}×{height})"
        if isinstance(parameter, Tensor):
            return f"Tensor({parameter.shape})"
        if isinstance(parameter, float):
            return f"{parameter:.02g}"
        if isinstance(parameter, dict):
            return "{" + ", ".join([
                f"{key} = {self.format_parameter(parameter[key])}"
                for key in parameter
            ])
        if isinstance(parameter, tuple):
            return "(" + ", ".join([
                self.format_parameter(part)
                for part in parameter
            ]) + ")"
        if isinstance(parameter, list):
            return "[" + ", ".join([
                self.format_parameter(part)
                for part in parameter
            ]) + "]"
        return Serializer.serialize(parameter)

    def format_parameters(self, parameters: Dict[str, Any]) -> str:
        """
        Formats a parameter dictionary into a string
        """
        return ", ".join([
            f"{key} = {self.format_parameter(parameters[key])}"
            for key in parameters
        ])

    def collage(
        self,
        results: List[Tuple[Dict[str, Any], Optional[str], List[Image]]]
    ) -> Union[Image, List[Image]]:
        """
        Builds the results into a collage.
        """
        from enfugue.util import fit_image
        from PIL import Image, ImageDraw
        
        # Get total images
        if self.use_video:
            total_images = len(results)
        else:
            total_images = sum([len(images) for kwargs, label, images in results])

        if total_images == 0:
            raise RuntimeError("No images passed.")

        # Get the number of rows and columns
        rows = total_images // self.grid_columns
        if total_images % self.grid_columns != 0:
            rows += 1

        columns = total_images % self.grid_columns if total_images < self.grid_columns else self.grid_columns

        # Calculate image height based on rows and columns
        width = self.grid_size * columns
        height = (self.grid_size * rows) + (self.caption_height * rows)

        # Create blank image
        grid = Image.new("RGB", (width, height), (255, 255, 255))

        # Multiply if making a video
        if self.use_video:
            frame_count = max([len(images) for kwargs, label, images in results])
            grid = [grid.copy() for i in range(frame_count)]
            draw = [ImageDraw.Draw(image) for image in grid]
        else:
            draw = ImageDraw.Draw(grid)

        # Iterate through each result image and paste
        row, column = 0, 0
        for parameter_set, label, images in results:
            for i, image in enumerate(images):
                # Fit the image to the grid size
                width, height = image.size
                image = fit_image(image, self.grid_size, self.grid_size, "contain", "center-center")
                # Figure out which image/draw to use
                if self.use_video:
                    target_image = grid[i]
                    target_draw = draw[i]
                else:
                    target_image = grid
                    target_draw = draw
                # Paste the image on the grid
                target_image.paste(
                    image,
                    (column * self.grid_size, row * (self.grid_size + self.caption_height))
                )
                # Put the caption under the image
                if label is None:
                    if self.use_video:
                        label = f"{self.format_parameters(parameter_set)}, {width}×{height}"
                    else:
                        label = f"{self.format_parameters(parameter_set)}, sample {i+1}, {width}×{height}"
                target_draw.text(
                    (column * self.grid_size + 5, row * (self.grid_size + self.caption_height) + self.grid_size + 2),
                    self.split_text(label),
                    fill=(0,0,0),
                    font=self.font
                )
                # Increment as necessary
                if not self.use_video:
                    column += 1
                    if column >= self.grid_columns:
                        row += 1
                        column = 0
            # Increment as necessary
            if self.use_video:
                column += 1
                if column >= self.grid_columns:
                    row += 1
                    column = 0
        return grid

    def execute(
        self,
        manager: DiffusionPipelineManager,
        *parameter_sets: Dict[str, Any]
    )-> Union[Image, Tuple[Image]]:
        """
        Executes each parameter set and pastes on the grid.
        """
        results: List[Tuple[Dict[str, Any], Optional[str], List[Image]]] = []
        for parameter_set in parameter_sets:
            manager.seed = self.seed
            label = parameter_set.pop("label", None)
            result = manager(**{**self.base_kwargs, **parameter_set})
            results.append((
                parameter_set,
                label,
                result["images"]
            ))
        return self.collage(results)
