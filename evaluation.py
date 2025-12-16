import os
import h5py
import argparse
import warnings
import numpy as np
import astropy.units as u
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from scipy.signal import fftconvolve
from astropy.coordinates import SkyCoord
from utils import do_aperture_photometry
from photutils.centroids import centroid_sources, centroid_com
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)    # Astropy throws fixing errors that can be ignored

parser = argparse.ArgumentParser(description='SNR and CNR evaluation on denoised images')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where denoised files live.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v/n2n]. Defualt: n2v')
args = parser.parse_args()


directory = Path(args.data_directory)
file_list = sorted(directory.glob(f'{args.denoising_model}*.fits'))


cnr_values  = []
snr_values  = []
fwhm_values = []


for file in file_list:
    science      = fits.open(file)
    science_data = science[0].data.astype(float)
    science_wcs  = WCS(science[0].header)
    if 'CTYPE1' not in science[0].header:   # Some files don't have celestial coordinates due to technical errors
        continue

    target_ra    = science[0].header['RA']
    target_dec   = science[0].header['DEC']
    target_coord = SkyCoord(ra=target_ra, dec=target_dec, frame='icrs', unit=(u.hourangle, u.deg))

    x, y     = target_coord.to_pixel(science_wcs)
    position = centroid_sources(science_data - np.median(science_data), xpos=x, ypos=y, box_size=25, centroid_func=centroid_com)
    position = (int(position[0][0]), int(position[1][0]))

    # Photometry parameters inferred by first looking at data
    photometry_radius = 12
    annulus_radius    = 18
    annulus_width     = 4

    snr, cnr, fwhm = do_aperture_photometry(science_data, position, photometry_radius, annulus_radius, annulus_width)
    
    snr_values.append(snr)
    cnr_values.append(cnr)
    fwhm_values.append(fwhm)
    

array1 = fits.getdata(file_list[0]).astype(float) 
array2 = fits.getdata(file_list[1]).astype(float) 

cross_corr = fftconvolve(array1, array2[::-1, ::-1], mode='full')

# Overwrite existing h5 file if it exists
with h5py.File(f'Results/{args.denoising_model}_data.h5', 'w') as f:
    f.create_dataset('snr', data=np.array(snr_values))
    f.create_dataset('cnr', data=np.array(cnr_values))
    f.create_dataset('fwhm', data=np.array(fwhm_values))
    f.create_dataset('cross_correlation', data=cross_corr)