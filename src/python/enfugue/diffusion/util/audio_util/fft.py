# adapted from https://github.com/aiXander/Realtime_PyAudio_FFT/blob/master/src/fft.py
import numpy as np

def getFFT(data, rate, chunk_size, log_scale=False):
    data = data * np.hamming(len(data))
    try:
        FFT = np.abs(np.fft.rfft(data)[1:])
    except:
        FFT = np.fft.fft(data)
        left, right = np.split(np.abs(FFT), 2)
        FFT = np.add(left, right[::-1])

    if log_scale:
        try:
            FFT = np.multiply(20, np.log10(FFT))
        except Exception as e:
            print('Log(FFT) failed: %s' %str(e))

    return FFT
