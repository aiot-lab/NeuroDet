#############################################
# Author: Luca Yu
# Date: 2024-08-12
# Description: This file contains the radar processing functions
#############################################

import numpy as np

def range_fft(data):
    return np.fft.fft(data,axis=-1)

def form_virtual_antennas(data, isAzimuthOnly=True):
    if isAzimuthOnly:
        # remove the TX2 channel
        data = data[:, [0, 2], :, :]
        data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
        num_virtual_antennas = data.shape[2]
    else:
        data = np.reshape(data, (data.shape[0], -1, data.shape[-1]))
        num_virtual_antennas = data.shape[2]
    return data, num_virtual_antennas

def Bartlett_doa_estimation(data ,azimuth_degree_range=(-60, 60)):
    M = data.shape[1]  # Total antenna elements = TX * RX
    azimuth_angles = np.deg2rad(np.arange(*azimuth_degree_range))
    steering_vectors = np.exp(1j * np.pi * np.outer(np.arange(M), np.sin(azimuth_angles)))
    beamformed_spectrum = np.einsum('fmr,ma->far', data, steering_vectors)
    return beamformed_spectrum

def processing(adc_data):
    range_spectrum = range_fft(adc_data)
    virtual_antennas_data, _ = form_virtual_antennas(range_spectrum)
    aoa_spectrogram = Bartlett_doa_estimation(virtual_antennas_data)
    return aoa_spectrogram