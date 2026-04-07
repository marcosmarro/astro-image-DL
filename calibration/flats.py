#!/usr/bin/env python3
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

def create_median_flat(
    flat_list: list,
    median_bias: np.ndarray,
    median_dark: np.ndarray,
    dark_exptime: float,
) -> np.ndarray:
    """
    Creates a reduced median dark frame array.

    Reads in a list of filepaths to dark frames, where the median_dark frame is subtracted
    from each flat image (remembering to scale the dark frame by the exposure time of the 
    flat frame). It is subtracted out by a median bias array if user inputs bias frames.
    It is then sigma clipped and returned as normalized flat array.

    Parameters
    ----------
    flat_list:    list
                  list of filepaths to flat images.
    median_bias:  ndarray
                  2D array of reduced median bias image.
    median_dark:  ndarray
                  2D array of reduced median dark image.
    dark_exptime: float
                  mean exposure time of dark frames.

    Returns
    -------
    median_flat: ndarray
                 2D array of reduced median flat image.
    """
    flat_bias_data = []
    
    # Will read each file and append to dark_bias_data list where the arrays have dtype = float32
    for file in flat_list:
        flat_img     = fits.open(file)
        flat_data    = flat_img[0].data[96:-96, 96:-96].astype('f4')
        flat_exptime = flat_img[0].header['EXPTIME']
        
        # Subtracts bias from each flat and adds to flat_bias_data list
        if median_bias is not None:
            flat_data -= median_bias
        
        flat_data -= median_dark * flat_exptime / dark_exptime
        
        flat_bias_data.append(flat_data)  

    # Reads the list of flats and sigma clips the arrays
    flat_sc = sigma_clip(flat_bias_data, cenfunc='median', sigma=3, axis=0)
    
    # Creates a final 2D array that is the mean of each pixel from all different flats, and then divides by the median 
    # flat value to normalize
    flat = np.ma.mean(flat_sc, axis=0)

    # Normalizes the resulting flat to get the median flat
    median_flat = flat / np.median(flat).data

    return median_flat