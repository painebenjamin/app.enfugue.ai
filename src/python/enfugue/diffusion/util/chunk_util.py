from dataclasses import dataclass
from typing import Optional, Iterator, List, Tuple, Union
from math import ceil

__all__ = ["Chunker"]

@dataclass
class Chunker:
    width: int
    height: int
    frames: Optional[int] = None
    size: Optional[Union[int, Tuple[int, int]]] = None
    stride: Optional[Union[int, Tuple[int, int]]] = None
    frame_size: Optional[int] = None
    frame_stride: Optional[int] = None
    tile: Union[bool, Tuple[bool, bool]] = False
    loop: bool = False
    vae_scale_factor: int = 8
    temporal_first: bool = False

    def get_pixel_from_latent(self, chunk: List[int]) -> List[int]:
        """
        Turns latent chunk into pixel chunk
        """
        start = chunk[0]
        start_px = start * self.vae_scale_factor
        end = chunk[-1]
        end_px = end * self.vae_scale_factor
        low = min(chunk)
        high = max(chunk)
        wrapped = start != low
        if wrapped:
            low_px = low * self.vae_scale_factor
            high_px = high * self.vae_scale_factor
            return list(range(low_px, high_px)) + list(range(end_px))
        return list(range(start_px, end_px))

    @property
    def latent_width(self) -> int:
        """
        Returns latent (not pixel) width
        """
        return self.width // self.vae_scale_factor

    @property
    def latent_height(self) -> int:
        """
        Returns latent (not pixel) height
        """
        return self.height // self.vae_scale_factor

    @property
    def latent_size(self) -> Tuple[int, int]:
        """
        Returns latent (not pixel) size
        """
        if self.size is None:
            return (self.latent_width, self.latent_height)
        if isinstance(self.size, tuple):
            width, height = self.size
        else:
            width, height = self.size, self.size
        return (
            width // self.vae_scale_factor,
            height // self.vae_scale_factor,
        )

    @property
    def latent_stride(self) -> Tuple[int, int]:
        """
        Returns latent (not pixel) stride
        """
        if self.stride is None:
            return (self.latent_width, self.latent_height)
        if isinstance(self.stride, tuple):
            left, top = self.stride
        else:
            left, top = self.stride, self.stride
        return (
            left // self.vae_scale_factor,
            top // self.vae_scale_factor,
        )

    @property
    def num_horizontal_chunks(self) -> int:
        """
        Gets the number of horizontal chunks.
        """
        if not self.size or not self.stride:
            return 1
        if isinstance(self.tile, tuple):
            tile_x, tile_y, = self.tile
        else:
            tile_x = self.tile
        if tile_x:
            return max(ceil(self.latent_width / self.latent_stride[0]), 1)
        return max(ceil((self.latent_width - self.latent_size[0]) / self.latent_stride[0] + 1), 1)

    @property
    def num_vertical_chunks(self) -> int:
        """
        Gets the number of vertical chunks.
        """
        if not self.size or not self.stride:
            return 1
        if isinstance(self.tile, tuple):
            tile_x, tile_y, = self.tile
        else:
            tile_y = self.tile
        if tile_y:
            return max(ceil(self.latent_height / self.latent_stride[1]), 1)
        return max(ceil((self.latent_height - self.latent_size[1]) / self.latent_stride[1] + 1), 1)

    @property
    def num_chunks(self) -> int:
        """
        Gets the number of latent space image chunks
        """
        return self.num_horizontal_chunks * self.num_vertical_chunks

    @property
    def num_frame_chunks(self) -> int:
        """
        Gets the number of frame chunks.
        """
        if not self.frames or not self.frame_size or not self.frame_stride:
            return 1
        if self.loop:
            return max(ceil(self.frames / self.frame_stride), 1)
        return max(ceil((self.frames - self.frame_size) / self.frame_stride + 1), 1)

    @property
    def tile_x(self) -> bool:
        """
        Gets whether or not tiling is eanbled on the X dimension.
        """
        if isinstance(self.tile, tuple):
            return self.tile[0]
        return self.tile

    @property
    def tile_y(self) -> bool:
        """
        Gets whether or not tiling is eanbled on the Y dimension.
        """
        if isinstance(self.tile, tuple):
            return self.tile[1]
        return self.tile

    @property
    def chunks(self) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Gets the chunked latent indices
        """
        if not self.size or not self.stride:
            yield (
                (0, self.latent_height),
                (0, self.latent_width)
            )
            return

        vertical_chunks = self.num_vertical_chunks
        horizontal_chunks = self.num_horizontal_chunks
        total = vertical_chunks * horizontal_chunks

        latent_size_x, latent_size_y = self.latent_size
        latent_stride_x, latent_stride_y = self.latent_stride
        if isinstance(self.tile, tuple):
            tile_x, tile_y = self.tile
        else:
            tile_x, tile_y = self.tile, self.tile

        for i in range(total):
            vertical_offset = None
            horizontal_offset = None

            top = (i // horizontal_chunks) * latent_stride_y
            bottom = top + latent_size_y

            left = (i % horizontal_chunks) * latent_stride_x
            right = left + latent_size_x

            if bottom > self.latent_height:
                vertical_offset = bottom - self.latent_height
                bottom -= vertical_offset
                if not tile_y:
                    top = max(0, top - vertical_offset)

            if right > self.latent_width:
                horizontal_offset = right - self.latent_width
                right -= horizontal_offset
                if not tile_x:
                    left = max(0, left - horizontal_offset)

            horizontal = [left, right]
            vertical = [top, bottom]

            if horizontal_offset is not None and tile_x:
               horizontal[-1] = horizontal_offset

            if vertical_offset is not None and tile_y:
                vertical[-1] = vertical_offset

            yield tuple(vertical), tuple(horizontal) # type: ignore

    @property
    def frame_chunks(self) -> Iterator[Tuple[int, int]]:
        """
        Iterates over the frame chunks.
        """
        if not self.frames:
            return
        if not self.frame_size or not self.frame_stride:
            yield (0, self.frames)
            return
        for i in range(self.num_frame_chunks):
            offset = None
            start = i * self.frame_stride
            end = start + self.frame_size

            if end > self.frames:
                offset = end - self.frames
                end -= offset
                if not self.loop:
                    start -= offset
            frames = [start, end]
            if offset is not None and self.loop:
                frames[-1] = offset

            yield tuple(frames) # type: ignore

    def __len__(self) -> int:
        """
        Implements len() to return the total number of chunks
        """
        return self.num_chunks * self.num_frame_chunks

    def __iter__(self) -> Iterator[
        Tuple[
            Tuple[int, int],
            Tuple[int, int], 
            Tuple[Optional[int], Optional[int]]
        ]
    ]:
        """
        Iterates over all chunks, yielding (vertical, horizontal, temporal)
        """
        if self.frames:
            if self.temporal_first:
                for frame_chunk in self.frame_chunks:
                    for vertical_chunk, horizontal_chunk in self.chunks:
                        yield (vertical_chunk, horizontal_chunk, frame_chunk)
            else:
                for vertical_chunk, horizontal_chunk in self.chunks:
                    for frame_chunk in self.frame_chunks:
                        yield (vertical_chunk, horizontal_chunk, frame_chunk)
        else:
            for vertical_chunk, horizontal_chunk in self.chunks:
                yield (vertical_chunk, horizontal_chunk, (None, None))
