# type: ignore
# Adapted from https://github.com/aiXander/Realtime_PyAudio_FFT/blob/master/src/stream_analyzer.py
import numpy as np
from collections import deque
from scipy.signal import savgol_filter

from enfugue.diffusion.util.audio_util.fft import getFFT
from enfugue.diffusion.util.audio_util.utils import *

class StreamAnalyzer:
    """
    The Audio_Analyzer class provides access to continuously recorded
    (and mathematically processed) audio data.

    Arguments:
        fft_window_size_ms: int:  Time window size (in ms) to use for the FFT transform
        updatesPerSecond: int:    How often to record new data.

    """

    def __init__(
        self,
        rate: float,
        fft_window_size: int,
        smoothing_length_ms = 50,
        n_frequency_bins    = 256,
    ):
        self.n_frequency_bins = n_frequency_bins
        self.rate = rate
        #Custom settings:
        self.rolling_stats_window_s    = 20     # The axis range of the FFT features will adapt dynamically using a window of N seconds
        self.equalizer_strength        = 0.20   # [0-1] --> gradually rescales all FFT features to have the same mean
        self.apply_frequency_smoothing = True   # Apply a postprocessing smoothing filter over the FFT outputs

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(0.03*self.n_frequency_bins) - 1

        self.fft_window_size = fft_window_size
        self.fft_window_size_ms = 1000 * self.fft_window_size / self.rate
        self.fft  = np.ones(int(self.fft_window_size/2), dtype=float)
        self.fftx = np.arange(int(self.fft_window_size/2), dtype=float) * self.rate / self.fft_window_size

        # Temporal smoothing:
        # Currently the buffer acts on the FFT_features (which are computed only occasionally eg 30 fps)
        # This is bad since the smoothing depends on how often the .get_audio_features() method is called...
        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = get_smoothing_filter(self.fft_window_size_ms, self.smoothing_length_ms)
            self.feature_buffer = numpy_data_buffer(len(self.smoothing_kernel), len(self.fft), dtype = np.float32, data_dimensions = 2)

        #This can probably be done more elegantly...
        self.fftx_bin_indices = np.logspace(np.log2(len(self.fftx)), 0, len(self.fftx), endpoint=True, base=2, dtype=None) - 1
        self.fftx_bin_indices = np.round(((self.fftx_bin_indices - np.max(self.fftx_bin_indices))*-1) / (len(self.fftx) / self.n_frequency_bins),0).astype(int)
        self.fftx_bin_indices = np.minimum(np.arange(len(self.fftx_bin_indices)), self.fftx_bin_indices - np.min(self.fftx_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres  = np.zeros(self.n_frequency_bins)
        self.fftx_indices_per_bin   = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(self.fftx_bin_indices == bin_index)
            self.fftx_indices_per_bin.append(bin_frequency_indices)
            fftx_frequencies_this_bin = self.fftx[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(fftx_frequencies_this_bin)

        #Hardcoded parameters:
        self.fft_fps = 30
        self.log_features = False   # Plot log(FFT features) instead of FFT features --> usually pretty bad
        self.delays = deque(maxlen=20)
        self.num_ffts = 0
        self.strongest_frequency = 0

        #Assume the incoming sound follows a pink noise spectrum:
        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(np.log2(self.rate/2)), len(self.fftx), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * self.fft_fps #Assumes ~30 FFT features per second
        self.rolling_bin_values = numpy_data_buffer(self.rolling_stats_window_n, self.n_frequency_bins, start_value = 25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values  = np.mean(self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values  = np.maximum((1-self.equalizer_strength)*np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self, latest_data_window):
        self.fft = getFFT(latest_data_window, self.rate, self.fft_window_size, log_scale = self.log_features)
        #Equalize pink noise spectrum falloff:
        self.fft = self.fft * self.power_normalization_coefficients
        self.num_ffts += 1

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.fft)
            buffered_features = self.feature_buffer.get_most_recent(len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.fft = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.fftx[np.argmax(self.fft)]

        #ToDo: replace this for-loop with pure numpy code
        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(self.fft[self.fftx_indices_per_bin[bin_index]])

        #Beat detection ToDo:
        #https://www.parallelcube.com/2018/03/30/beat-detection-algorithm/
        #https://github.com/shunfu/python-beat-detector
        #https://pypi.org/project/vamp/

        return

    def __call__(self, data_window):
        self.update_features(data_window)
        self.update_rolling_stats()

        self.frequency_bin_energies = np.nan_to_num(self.frequency_bin_energies, copy=True)
        if self.apply_frequency_smoothing:
            if self.filter_width > 3:
                self.frequency_bin_energies = savgol_filter(self.frequency_bin_energies, self.filter_width, 3)
        self.frequency_bin_energies /= self.bin_mean_values
        self.frequency_bin_energies[np.isnan(self.frequency_bin_energies)] = 0
        self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0
        self.frequency_bin_energies[self.frequency_bin_energies > 1] = 1
        return self.fftx, self.fft, self.frequency_bin_centres, self.frequency_bin_energies
