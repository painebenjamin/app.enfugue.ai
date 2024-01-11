# adapted from https://github.com/aiXander/Realtime_PyAudio_FFT/blob/master/src/utils.py
import numpy as np
import math, scipy

def round_up_to_even(f):
    return int(math.ceil(f / 2.) * 2)

def round_to_nearest_power_of_two(f, base=2):
    l = math.log(f,base)
    rounded = int(np.round(l,0))
    return base**rounded

def get_frequency_bins(start, stop, n):
    octaves = np.logspace(log(start)/log(2), log(stop)/log(2), n, endpoint=True, base=2, dtype=None)
    return np.insert(octaves, 0, 0)

def gaussian_kernel1d(sigma, truncate=2.0):
    sigma = float(sigma)
    sigma2 = sigma * sigma
    # make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sigma + 0.5)
    exponent_range = np.arange(1)

    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

def gaussian_kernel_1D(w, sigma):
    sigma = sigma
    x = np.linspace(-sigma, sigma, w+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    return kern1d/kern1d.sum()

def get_smoothing_filter(FFT_window_size_ms, filter_length_ms):
    buffer_length = round_up_to_even(filter_length_ms / FFT_window_size_ms)+1
    filter_sigma = buffer_length / 3  #How quickly the smoothing influence drops over the buffer length
    filter_weights = gaussian_kernel1d(filter_sigma)[:,np.newaxis]

    max_index = np.argmax(filter_weights)
    filter_weights = filter_weights[:max_index+1]
    filter_weights = filter_weights / np.mean(filter_weights)

    return filter_weights

class numpy_data_buffer:
    """
    A fast, circular FIFO buffer in numpy with minimal memory interactions by using an array of index pointers
    """

    def __init__(self, n_windows, samples_per_window, dtype = np.float32, start_value = 0, data_dimensions = 1):
        self.n_windows = n_windows
        self.data_dimensions = data_dimensions
        self.samples_per_window = samples_per_window
        self.data = start_value * np.ones((self.n_windows, self.samples_per_window), dtype = dtype)

        if self.data_dimensions == 1:
            self.total_samples = self.n_windows * self.samples_per_window
        else:
            self.total_samples = self.n_windows

        self.elements_in_buffer = 0
        self.overwrite_index = 0

        self.indices = np.arange(self.n_windows, dtype=np.int32)
        self.last_window_id = np.max(self.indices)
        self.index_order = np.argsort(self.indices)

    def append_data(self, data_window):
        self.data[self.overwrite_index, :] = data_window

        self.last_window_id += 1
        self.indices[self.overwrite_index] = self.last_window_id
        self.index_order = np.argsort(self.indices)

        self.overwrite_index += 1
        self.overwrite_index = self.overwrite_index % self.n_windows

        self.elements_in_buffer += 1
        self.elements_in_buffer = min(self.n_windows, self.elements_in_buffer)

    def get_most_recent(self, window_size):
        ordered_dataframe = self.data[self.index_order]
        if self.data_dimensions == 1:
            ordered_dataframe = np.hstack(ordered_dataframe)
        return ordered_dataframe[self.total_samples - window_size:]

    def get_buffer_data(self):
        return self.data[:self.elements_in_buffer]
