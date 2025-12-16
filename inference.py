import os
import gc
import torch
import argparse
import warnings
from tqdm import tqdm
from pathlib import Path
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)    # Astropy throws fixing errors that can be ignored

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ML training script on astronomical images.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v/n2n]. Defualt: n2v')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where science files live.')
args = parser.parse_args()

batchsize = 1

if args.denoising_model == 'n2v':
    model = torch.load('n2v.pth', weights_only=False)
    model.eval()

if args.denoising_model == 'n2n':
    model = torch.load('n2n.pth', weights_only=False)
    model.eval()

directory = Path(args.data_directory)
file_list = sorted(directory.glob('*.fits'))


for i, file in tqdm(enumerate(file_list)):

    # Reading data from files, rememebring to discard outer pixels due to overscan regions
    science      = fits.open(file)
    science_data = science[0].data.astype(float)

    train_loader = torch.from_numpy(science_data).float().to(DEVICE)
    train_loader = train_loader.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        output_sequence = model(train_loader)
    
    denoised = output_sequence.squeeze().cpu().numpy()
    
    # Writing the data into the file where it's saved in directory named DenoisedScience
    science_hdu = fits.PrimaryHDU(data=denoised, header=science[0].header)
    hdu_list    = fits.HDUList([science_hdu])

    hdu_list.writeto(f'Denoised_Science/{args.denoising_model}_{file.stem}.fits', overwrite=True)

    print(f'Denoised {file}')
    del file, train_loader
    gc.collect()