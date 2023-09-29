"""
Some helper functions used in the experiment
@authors: GA and LS
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import curve_fit


def save_mask(noise, sp, file_name):
    save_dict = {
        "noise": noise,
        "stimulus_params": sp,
        }

    with open(file_name, 'wb') as handle:
        pickle.dump(save_dict, handle)


def load_mask(file_name):
    # Load data from pickle:
    with open(file_name, 'rb') as handle:
        data_pickle = pickle.load(handle)
    return data_pickle["noise"], data_pickle["stimulus_params"]


def make_sound(f, sampleRate=44100):
    x = np.arange(0, sampleRate)
    arr = (4096 * np.sin(2.0 * np.pi * f * x / sampleRate)).astype(np.int16)
    return arr


def create_directory(dir_path):
    if os.path.exists(dir_path) == 1:
        raise NameError('Directory exists. Change path or delete existing folder.')
    else:
        os.makedirs(dir_path)


# Define a line function to fit the lut
def line(var, a, b):
    return a * var + b


# Fit a line to lut values and return line params
def fit_line(lut):
    # Read out intensity and luminance values from lut:
    df = pd.read_csv(lut, sep=' ')
    intensity = df['IntensityIn']
    lum = df['Luminance']

    # Fit a line to values
    line_params, _ = curve_fit(line, intensity, lum)
    a = line_params[0]
    b = line_params[1]
    return a, b


# Translate intensity values to luminance based on fitted lut
def intensity_to_lum(intensity, a, b):
    lum = line(intensity, a, b)
    return lum


# Translate luminance values to intensity based on fitted lut
def lum_to_intensity(lum, a, b):
    intensity = (lum - b) / a
    return intensity


# Class with lut parameters
class get_lut_params(object):
    def __init__(self, lut):
        a, b = fit_line(lut)
        self.a = a
        self.b = b
