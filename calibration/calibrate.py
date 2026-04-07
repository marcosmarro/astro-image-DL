#!/usr/bin/env python
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from calibration.bias import create_median_bias
from calibration.darks import create_median_dark
from calibration.flats import create_median_flat
from calibration.science import calibrate_science_frames

def calibrate_science_images(
    bias_files: list, 
    dark_files: list, 
    flat_files: list, 
    science_files: list, 
    output_dir: str = './',
) -> None:
    """
    Calibrate science FITS images by applying bias, dark, and flat corrections.
    Saves the calibrated images to the specified output directory.

    Parameters
    ----------
    bias_files:    list
                   list of paths to bias frame FITS files.
    dark_files:    list
                   list of paths to dark frame FITS files.
    flat_files:    list
                   list of paths to flat frame FITS files.
    science_files: list
                   list of paths to science FITS files.
    output_dir :   str
                   directory where calibrated images will be saved. Default is current directory.

    Returns
    -------
    None
    """
    # Assigning 2D median bias, dark, and flat arrays
    bias           = create_median_bias(bias_files)
    dark, exp_time = create_median_dark(dark_files, bias)        # Need dark exposure time for flat fame reduction
    flat           = create_median_flat(flat_files, bias, dark, exp_time)
    calibrate_science_frames(science_files, bias, dark, flat, output_dir)