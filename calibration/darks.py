#!/usr/bin/env python3
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

def create_median_dark(dark_list: list, median_bias: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Creates a reduced median dark frame array.

    Reads in a list of filepaths to dark frames and stacks them together to later 
    be sigma clipped and averaged out. Each frame is subtracted by a median bias array 
    if user inputs bias frames.

    Parameters
    ----------
    dark_list:   list
                 list of filepaths to dark images.
    median_bias: ndarray
                 2D array of reduced median bias image.
        
    Returns
    -------
    median dark:  ndarray
                  2D array of reduced median dark image.
    dark_exptime: float
                  mean exposure time of dark frames.
    """
    dark_images = []
    exp_times   = []

    # Will read each file and append to dark_bias_data list where the arrays have dtype = float32
    for file in dark_list:
        dark_img  = fits.open(file)
        dark_data = dark_img[0].data[96:-96, 96:-96].astype('f4')
        exptime   = dark_img[0].header['EXPTIME']
        exp_times.append(exptime)

        # Subtracts bias from each dark image
        if median_bias is not None:
            dark_data -= median_bias
        
        # Divides each dark image (with bias subtracted) by the exposure time to get dark current, and adds to list
        dark_images.append(dark_data / exptime) 

    # Reads the list of darks and sigma clips the arrays
    dark_sc = sigma_clip(dark_images, cenfunc='median', sigma=3, axis=0)

    # Creates a final 2D array that is the mean of each pixel from all different darks
    median_dark  = np.ma.mean(dark_sc, axis=0).data
    dark_exptime = np.mean(exp_times)

    return median_dark, dark_exptime