from dataclasses import dataclass
from typing import Optional, Iterator, List, Tuple
from math import ceil

__all__ = ["Chunker"]

@dataclass
class Chunker:
    width: int
    height: int
    frames: Optional[int] = None
    size: Optional[int] = None
    stride: Optional[int] = None
    frame_size: Optional[int] = None
    frame_stride: Optional[int] = None
    loop: bool = False
    tile: bool = False
    vae_scale_factor: int = 8

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
    def latent_size(self) -> int:
        """
        Returns latent (not pixel) size
        """
        if self.size is None:
            return max(self.latent_width, self.latent_height)
        return self.size // self.vae_scale_factor

    @property
    def latent_stride(self) -> int:
        """
        Returns latent (not pixel) stride
        """
        if self.stride is None:
            return max(self.latent_width, self.latent_height)
        return self.stride // self.vae_scale_factor

    @property
    def num_horizontal_chunks(self) -> int:
        """
        Gets the number of horizontal chunks.
        """
        if not self.size or not self.stride:
            return 1
        if self.tile:
            return ceil(self.latent_width / self.latent_stride)
        return ceil((self.latent_width - self.latent_size) / self.latent_stride + 1)

    @property
    def num_vertical_chunks(self) -> int:
        """
        Gets the number of vertical chunks.
        """
        if not self.size or not self.stride:
            return 1
        if self.tile:
            return ceil(self.latent_height / self.latent_stride)
        return ceil((self.latent_height - self.latent_size) / self.latent_stride + 1)

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
            return ceil(self.frames / self.frame_stride)
        return ceil((self.frames - self.frame_size) / self.frame_stride + 1)

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
        for i in range(total):
            vertical_offset = None
            horizontal_offset = None

            top = (i // horizontal_chunks) * self.latent_stride
            bottom = top + self.latent_size

            left = (i % horizontal_chunks) * self.latent_stride
            right = left + self.latent_size

            if bottom > self.latent_height:
                vertical_offset = bottom - self.latent_height
                bottom -= vertical_offset
                if not self.tile:
                    top -= vertical_offset

            if right > self.latent_width:
                horizontal_offset = right - self.latent_width
                right -= horizontal_offset
                if not self.tile:
                    left -= horizontal_offset

            horizontal = [left, right]
            vertical = [top, bottom]

            if horizontal_offset is not None and self.tile:
               horizontal[-1] = horizontal_offset

            if vertical_offset is not None and self.tile:
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

    def __iter__(self) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int], Tuple[Optional[int], Optional[int]]]]:
        """
        Iterates over all chunks, yielding (vertical, horizontal, temporal)
        """
        if self.frames:
            for frame_chunk in self.frame_chunks:
                for vertical_chunk, horizontal_chunk in self.chunks:
                    yield (vertical_chunk, horizontal_chunk, frame_chunk)
        else:
            for vertical_chunk, horizontal_chunk in self.chunks:
                yield (vertical_chunk, horizontal_chunk, (None, None))
