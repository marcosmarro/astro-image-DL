import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import fitting
from astropy.modeling.models import Gaussian2D
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry


def do_aperture_photometry(
    data: np.ndarray,
    position: tuple[int, int],
    radius: int,
    sky_radius_in: int,
    sky_annulus_width: int,
    RON: float = 17.26,
) -> tuple:
    """Performs aperture photometry and returns a tuple of values needed to calculate SNR/CNR.

    Parameters
    ----------
    data:          array
                   2D data of reduced science image.
    position:      tuple
                   tuple of position location with integers (x, y).
    radius:        int
                   aperture radius in pixels.
    sky_radius: int
                   pixel radius at which to measure the sky background.
    sky_annulus_width: pixel width of the annulus.
    RON: Read-out noise that was calculated prior to denoising.

    Returns
    -------
        A tuple[SNR, CNR] containing:
            SNR: signal-to-noise ratio.
            CNR: contrast-to-noise ratio.
    """
    x, y = position[0], position[1]
    
    # Makes a circular aperture and caclulates the total flux in the area
    aperture = CircularAperture(position, radius)
    raw_flux = aperture_photometry(data, aperture)['aperture_sum'][0]

    # Makes a circular annulus and calculates the total background flux in that area
    annulus        = CircularAnnulus(position, sky_radius_in, sky_radius_in + sky_annulus_width)
    raw_background = aperture_photometry(data, annulus)['aperture_sum'][0]

    # Grabs the background's mean in the annulus and multiplies it by aperture's area to grab background in only annulus
    mean_noise = (raw_background / annulus.area).item()
    noise      = data[y - 10: y + 10, x - 100: x - 50].flatten().flatten()
    noise_std  = np.std(noise)

    # Background count in the aperture
    background = mean_noise * aperture.area

    # Calculates total flux
    signal = raw_flux - background

    # Calculates CNR and SNR
    cnr = (signal - mean_noise) / noise_std
    snr = signal / np.sqrt(signal + aperture.area * (mean_noise + RON ** 2))

    ### Calculating FWHM
    sub    = data[y-radius:y+radius+1, x-radius:x+radius+1]
    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    
    # Initial guess based on moments
    amp_guess   = sub.max()
    xo, yo      = radius, radius
    sigma_guess = radius / 2

    # Fitting data to a 2D Gaussian
    p_init = Gaussian2D(amplitude=amp_guess, x_mean=xo, y_mean=yo,
                        x_stddev=sigma_guess, y_stddev=sigma_guess)
    fit = fitting.LevMarLSQFitter()
    p = fit(p_init, xx, yy, sub)

    # Calculating mean FWHM from FWHMx and FWHMy
    fwhm_x = 2.355 * np.abs(p.x_stddev.value)
    fwhm_y = 2.355 * np.abs(p.y_stddev.value)

    fwhm = (fwhm_x + fwhm_y) / 2

    return snr, cnr, fwhm


def plot_comparisons(models: list):
    """Plots SNR, CNR, and FWHM for different models.

    Args:
        models: list of strings of models wished to be plot
            - example: ['original', 'n2v', 'n2n', 'standard']
    """
    # Create SNR figure
    fig_snr, ax_snr = plt.subplots()
    ax_snr.grid(True)

    # Create CNR figure
    fig_cnr, ax_cnr = plt.subplots()
    ax_cnr.grid(True)

    fig_fwhm, ax_fwhm = plt.subplots()
    ax_fwhm.grid(True)

    for model in models:
        file = h5py.File(f'Results/{model}_data.h5', 'r')

        model_snr  = file['snr'][:]
        model_cnr  = file['cnr'][:]
        model_fwhm = file['fwhm'][:]
     
        ax_snr.plot(model_snr, label=model)
        ax_snr.set_xlabel('File number')
        ax_snr.set_ylabel('SNR')

        ax_cnr.plot(model_cnr, label=model)
        ax_cnr.set_xlabel('File number')
        ax_cnr.set_ylabel('CNR')

        ax_fwhm.plot(model_fwhm, label=model)
        ax_fwhm.set_xlabel('File number')
        ax_fwhm.set_ylabel('FWHM')

    # Add legends and save after all models are plotted
    ax_snr.legend()
    fig_snr.savefig("Results/SNR.pdf", dpi=300)
    plt.close(fig_snr)

    ax_cnr.legend()
    fig_cnr.savefig("Results/CNR.pdf", dpi=300)
    plt.close(fig_cnr)

    ax_fwhm.legend()
    fig_fwhm.savefig("Results/FWHM.pdf", dpi=300)
    plt.close(fig_fwhm)


def plot_cross_correlation(models: list):
    """Plots SNR, CNR, and FWHM for different models.

    Args:
        models: list of strings of models wished to be plot
            - example: ['original', 'n2v', 'n2n', 'standard']
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid
    axes = axes.ravel()

    for i, model in enumerate(models):  # loop over 4 plots
        file = h5py.File(f'Results/{model}_data.h5', 'r')

        cross_corr = file['cross_correlation'][:]
        
        N = cross_corr.shape[0] // 2
        lags = np.arange(-N+1, N)

        im = axes[i].imshow(
            cross_corr, cmap='viridis', origin='lower',
            extent=[lags[0], lags[-1], lags[0], lags[-1]]
        )
        axes[i].set_title(f'{model} cross correlation')
        axes[i].set_xlabel('Lag X')
        axes[i].set_ylabel('Lag Y')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('Results/cross_correlation.pdf', dpi=300)
    plt.close()


def plot_sample(input: torch.Tensor, output: torch.Tensor):
    """Plots a denoising sample in current directory.

    Args:
        input: input sequence
        output: model's output sequence
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(input[0, 0].detach().cpu().numpy(), cmap="gray")
    axs[0].set_title("Noisy Input")
    axs[0].axis("off")

    axs[1].imshow(output[0, 0].detach().cpu().numpy(), cmap="gray")
    axs[1].set_title("Denoised Output")
    axs[1].axis("off")

    plt.savefig('denoise_sample.pdf', dpi=300)
    plt.close()