#!/usr/bin/env python
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astroscrappy import detect_cosmics

def calibrate_science_frames(
    science_list: list,
    median_bias: np.ndarray,
    median_dark: np.ndarray,
    median_flat: np.ndarray,
    output_dir: str = './',
) -> None:
    """
    Reduces a list of science frames and writes each to a new file.

    Reads in a list of filepaths to science frames to be reduced.
    It goes through standard astrophotography correction process given
    median bias, dark, and flat frames.

    .. Follows the equation:: Reduced = (Science - Expsoure time * Dark - Bias) / Flat

    It is then cosmic ray corrected to return final reduced frame.

    Parameters
    ----------
    science_list: list
                    list of file paths to science frames to be reduced.
    median_bias:  ndarray
                    2D array of reduced median bias image.
    median_dark:  ndarray
                    2D array of reduced median dark image.
    median_flat:  ndarray
                    2D array of reduced median flat image.
    output_dir:   str
                    directory to write the reduced files

    Returns
    -------
    None
    """
    reduced_paths = []

    for science_file in science_list:
        basename  = os.path.basename(science_file)
        reduced_filename = f"calibrated_{basename}"
        reduced_filepath = os.path.join(output_dir, reduced_filename)
        
        # Reads all the files and grabs their respective array
        science = fits.open(science_file)

        # Slices data array to tranfer all of the header information to the reduced file
        original_header = science[0].header
        original_wcs    = WCS(original_header)
        cropped_wcs     = original_wcs.slice((slice(96, -96), slice(96, -96)))
        cropped_header  = cropped_wcs.to_header()

        for key in original_header:
            if key in ('HISTORY', 'COMMENT'):
                continue
            if key not in cropped_header:
                cropped_header[key] = original_header[key]

        science_data  = science[0].data[96:-96, 96:-96].astype('f4')

        # Exposure time of science to later use with median dark 
        exposure_time = science[0].header['EXPTIME']

        # Removes bias is user inputs bias image
        if median_bias is not None:
            science_data -= median_bias 

        # Removes dark frame and divides out by flat image
        science_data -= exposure_time * median_dark
        science_data /= median_flat

        # Removal of cosmic rays
        mask, cleaned   = detect_cosmics(science_data)
        reduced_science = cleaned

        # Create a new FITS file from the resulting reduced science frame.
        science_hdu = fits.PrimaryHDU(data=reduced_science.data, header=cropped_header)
        science_hdu.header['COMMENT'] = "Reduced science image correcting from inputed frames."
        hdul = fits.HDUList([science_hdu])
        hdul.writeto(reduced_filepath, overwrite=True)

        reduced_paths.append(reduced_filepath)