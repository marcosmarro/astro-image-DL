#!/usr/bin/env python3
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip

def create_median_bias(bias_list: list) -> np.ndarray:
    """
    Creates a reduced median bias array.

    Reads in a list of filepaths to bias frames and stacks them together to later 
    be sigma clipped and averaged out.

    Note that this function is completely optional as the user is not required to 
    input bias files.

    Parameters
    ----------
    bias_list: list
               list of filepaths to bias images.
        
    Returns
    -------
    median_bias: ndarray
                 2D array of reduced median bias image.
    """
    # If the user inputs bias image(s), then the script will run
    if bias_list:
        bias_images = []

        # Will read each file and append to bias_images list where the arrays have dtype = float32
        for bias in bias_list:
            bias_data = fits.getdata(bias)[96:-96, 96:-96]
            bias_images.append(bias_data.astype('f4'))

        # Reads the list of biases and sigma clips the arrays
        bias_images_masked = sigma_clip(bias_images, cenfunc='median', sigma=3, axis=0) 

        # Creates a final 2D array that is the mean of each pixel from all different biases
        median_bias = np.ma.mean(bias_images_masked, axis=0).data

        return median_bias

    else:
        pass # Return none if user doesn't input bias files