from __future__ import annotations

import os

from enfugue.util import reiterator
from typing import TYPE_CHECKING, Iterable, Tuple, Optional, List, Dict, Callable

if TYPE_CHECKING:
    from moviepy.editor import (
        AudioClip,
        AudioFileClip,
        CompositeAudioClip
    )

__all__ = ["Audio"]

class Audio:
    def __init__(
        self,
        frames: Iterable[Tuple[float]],
        rate: Optional[int]=None
    ) -> None:
        self.frames = reiterator(frames)
        self.rate = rate

    def get_clip(self, rate: Optional[int]=None) -> AudioClip:
        """
        Gets the moviepy audioclip
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100

        from moviepy.editor import AudioClip

        all_frames = [frame for frame in self.frames] # type: ignore
        total_frames = len(all_frames)
        duration = total_frames / rate

        def get_frame(time: float) -> Tuple[float]:
            if isinstance(time, int) or isinstance(time, float):
                return all_frames[int(total_frames*time)]
            return [ # type: ignore[unreachable]
                all_frames[int(t*rate)]
                for t in time
            ]

        return AudioClip(get_frame, duration=duration, fps=rate)

    def get_composite_clip(self, rate: Optional[int]=None) -> CompositeAudioClip:
        """
        Gets the moviepy composite audioclip
        """
        from moviepy.editor import CompositeAudioClip
        return CompositeAudioClip([self.get_clip(rate=rate)])

    def save(
        self,
        path: str,
        rate: Optional[int]=None
     ) -> int:
        """
        Saves the audio frames to file
        """
        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100
        if path.startswith("~"):
            path = os.path.expanduser(path)
        clip = self.get_clip(rate=rate)
        clip.write_audiofile(path)
        if not os.path.exists(path):
            raise IOError(f"Nothing was written to {path}.")
        size = os.path.getsize(path)
        if size == 0:
            raise IOError(f"Nothing was written to {path}.")
        return size

    def get_frequencies(
        self,
        samples_per_second: int=60,
        maximum_samples: Optional[int]=None,
        rate: Optional[int]=None,
    ) -> Iterable[Dict[int, float]]:
        """
        Gets channel frequencies over the audio frames using RFFT
        """
        import numpy as np
        from math import sqrt

        if not rate:
            rate = self.rate
        if not rate:
            rate = 44100

        samples_per_yield = int(rate / samples_per_second)
        samples: List[Tuple[float]] = [(0.0,)] * samples_per_yield
        current_samples = 0
        yielded_samples = 0

        def get_frequencies() -> Dict[int, float]:
            num_channels = len(samples[0])
            channel_frequencies = []
            for i in range(num_channels):
                these_samples = [
                    samples[j][i]
                    for j in range(current_samples)
                ]
                channel_frequencies.append([
                    sqrt(f.real * f.real + f.imag * f.imag) for f in np.fft.rfft(these_samples)
                ])
            xf = np.fft.rfftfreq(current_samples, d=1/rate) # type: ignore[operator]
            return dict(zip(
                [int(f) for f in xf],
                [list(f) for f in zip(*channel_frequencies)] # type: ignore[misc]
            ))

        for frame in self.frames: # type: ignore[attr-defined]
            samples[current_samples] = frame
            current_samples += 1
            if current_samples >= samples_per_yield:
                yield get_frequencies()
                yielded_samples += 1
                current_samples = 0
                if maximum_samples is not None and yielded_samples >= maximum_samples:
                    break
        if current_samples != 0:
            if maximum_samples is None or yielded_samples < maximum_samples:
                yield get_frequencies()

    @classmethod
    def frequencies_from_file(
        cls,
        path: str,
        skip_frames: Optional[int]=None,
        samples_per_second: int=8,
        maximum_samples: Optional[int]=None
   ) -> Iterable[Dict[int, float]]:
        """
        Gets frequencies over a file iterable
        """
        audio = cls.from_file(path, skip_frames=skip_frames)
        rate = audio.rate if audio.rate is not None else 44100
        for i, histogram in enumerate(audio.get_frequencies(rate, samples_per_second)):
            yield histogram
            if maximum_samples is not None and i >= maximum_samples:
                break

    @classmethod
    def file_to_frames(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Iterable[Tuple[float]]:
        """
        Starts an audio capture and yields tuples for each frame.
        """
        from moviepy.editor import AudioFileClip
        if path.startswith("~"):
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise IOError(f"Audio at path {path} not found or inaccessible")

        clip = AudioFileClip(path)
        if on_open is not None:
            on_open(clip)

        total_frames = 0
        for i, frame in enumerate(clip.iter_frames()):
            if skip_frames is not None and i < skip_frames:
                continue
            if maximum_frames is not None and total_frames + 1 > maximum_frames:
                break
            yield frame
            total_frames += 1

        if total_frames == 0:
            raise IOError(f"No frames were read from audio at path {path}")

    @classmethod
    def from_file(
        cls,
        path: str,
        skip_frames: Optional[int] = None,
        maximum_frames: Optional[int] = None,
        on_open: Optional[Callable[[AudioFileClip], None]] = None
    ) -> Audio:
        """
        Uses Audio.frames_from_file and instantiates an Audio object.
        """
        rate: Optional[int] = None

        def get_rate_on_open(clip: AudioFileClip) -> None:
            nonlocal rate
            rate = clip.fps
            if on_open is not None:
                on_open(clip)

        frames=cls.file_to_frames(
            path=path,
            skip_frames=skip_frames,
            maximum_frames=maximum_frames,
            on_open=get_rate_on_open
        )
        return cls(frames=frames, rate=rate)
