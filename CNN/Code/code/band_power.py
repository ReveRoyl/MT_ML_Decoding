import numpy as np
import math
from scipy.integrate import simps
from mne.time_frequency import psd_array_welch

def bandpower_1d(data, sf, band, nperseg=None, relative=False):
    """
        Compute the average power of the signal x in a specific frequency band.
        https://raphaelvallat.com/bandpower.html
    Args:
        data (1d-array):
            Input signal in the time-domain.
        sf (float):
            Sampling frequency of the data.
        band (list):
            Lower and upper frequencies of the band of interest.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (float):
            Absolute or relative band power.
    """

    # band = np.asarray(band)
    low, high = band

    # Compute the modified periodogram (Welch)
    psd, freqs = psd_array_welch(data, sf, 1., 70., n_per_seg=None,
                          n_overlap=0, n_jobs=1)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def bandpower(x, fs, fmin, fmax, nperseg=None, relative=True):
    """
    Compute the average power of the multi-channel signal x in a specific frequency band.
    Args:
        x (nd-array): [n_epoch, n_channel, n_times]
           The epoched input data.
        fs (float):
            Sampling frequency of the data.
        fmin (int): Low-band frequency.
        fmax (int): High-band frequency.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (nd-array): [n_epoch, n_channel, 1]
            Absolute or relative band power.
    """
    n_epoch, n_channel, _ = x.shape

    bp = np.zeros((n_epoch, n_channel, 1))
    for epoch in range(n_epoch):
        for channel in range(n_channel):
            bp[epoch, channel] = bandpower_1d(
                x[epoch, channel, :],
                fs,
                [fmin, fmax],
                nperseg=nperseg,
                relative=relative,
            )

    return bp

def bandpower_multi(x, fs, bands,  nperseg=None, relative=True):
    """
    Compute the average power of the multi-channel signal x in multiple frequency bands.
    Args:
        x (nd-array): [n_epoch, n_channel, n_times]
           The epoched input data.
        fs (float):
            Sampling frequency of the data.
        bands (list): list of bands to compute the bandpower. echa band is a tuple of fmin and fmax.
        window_sec (float):
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative (boolean):
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.

    Returns:
        bp (nd-array): [n_epoch, n_channel, n_bands]
            Absolute or relative bands power.
    """
    n_epoch, n_channel, _ = x.shape
    bp_list = []
    for idx, band in enumerate(bands):
        fmin, fmax = band
        bp_list.append(
            bandpower(
                x, fs, fmin, fmax,  nperseg=None, relative=relative
            )
        )

    bp = np.concatenate(bp_list, -1)

    return bp