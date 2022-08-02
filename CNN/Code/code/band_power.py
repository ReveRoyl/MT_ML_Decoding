import numpy as np
import math
import os
import sys
import mne
import torch
from mne.decoding import Scaler
from mne.decoding import UnsupervisedSpatialFilter
from mne.time_frequency import psd_array_welch
from numpy import trapz
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from scipy.signal import welch
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as skScaler
def bandpower_1d(data, sf, band, nperseg=800, relative=False):
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
    # TODO: generalize freq values
    psd, freqs = psd_array_welch(data, sf, 1., 70., n_per_seg=int(800 / 2),
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

def bandpower(x, fs, bands, nperseg=800, relative=True):
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

    #TODO fix nperseg

    psd, freqs = psd_array_welch(x, fs, 1., 70., n_per_seg=int(fs/2),
                                 n_overlap=0, n_jobs=1)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    n_channel, _ = x.shape
    bp = np.zeros((n_channel, len(bands)))
    for idx, band in enumerate(bands):
        low, high = band
        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        _bp = simps(psd[..., idx_band], dx=freq_res, axis=-1)

        if relative:
            _bp /= simps(psd, dx=freq_res, axis=-1)
        
        # print(bp.shape, _bp.shape) #272,6  80,272
        bp[:, idx] = _bp

    return bp

def bandpower_multi_bands(x, fs, bands,  nperseg=800, relative=True):
    
    n_epoch, n_channel, _ = x.shape
    bp = np.zeros((n_epoch, n_channel, len(bands)))
    for e in range(n_epoch):
        bp[e] = bandpower(x[e], fs, bands, nperseg=nperseg, relative=relative)

    return bp

# def bandpower_multi(x, fs, bands,  nperseg=100, relative=True):
#     """
#     Compute the average power of the multi-channel signal x in multiple frequency bands.
#     Args:
#         x (nd-array): [n_epoch, n_channel, n_times]
#            The epoched input data.
#         fs (float):
#             Sampling frequency of the data.
#         bands (list): list of bands to compute the bandpower. echa band is a tuple of fmin and fmax.
#         window_sec (float):
#             Length of each window in seconds.
#             If None, window_sec = (1 / min(band)) * 2
#         relative (boolean):
#             If True, return the relative power (= divided by the total power of the signal).
#             If False (default), return the absolute power.
#     Returns:
#         bp (nd-array): [n_epoch, n_channel, n_bands]
#             Absolute or relative bands power.
#     """
#     n_epoch, n_channel, _ = x.shape
#     bp_list = []
#     for idx, band in enumerate(bands):
#         fmin, fmax = band
#         bp_list.append(
#             bandpower(
#                 x, fs, bands=bands, relative=relative # nperseg=100,
#             )
#         )

#     bp = np.concatenate(bp_list, -1)

#     return bp

def standard_scaling_sklearn(data):
    """
    Standard scale the input data on last dimension. It center the data to 0 and scale to unit variance.
    It scales trial- or epoch-wise, estimating the mean and the std using the timepoints of a single trial.
    Args:
        data (nd-array): [n_epochs, n_channel, n_times]
            The data to scale.
    Returns:
        data (nd-array): [n_epochs, n_channel, n_times]
            The standardized data.
    """
    n_epoch = data.shape[0]
    for e in range(n_epoch):
        scaler = skScaler()
        data[e, ...] = scaler.fit_transform(data[e, ...])

    return data

def onehot(batches, n_classes, y):
  yn = torch.zeros(batches, n_classes)
  for i in range(batches):
    x = [0 for j in range(batches)]
    x[i] = y[i]/2-1                     #ex. [12]-> [5]
    yn[i][int(x[i])]+= 1                  #[000010000]
  return yn