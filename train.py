import os
import gc
import torch
import argparse
import torch.nn as nn
from pathlib import Path
from astropy.io import fits
from network import UNetN2V, UNetN2N
from utils import random_crop, apply_n2v_mask, plot_sample
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='ML training script on astronomical images.')
parser.add_argument('--denoising_model', '-m', default='n2v', help='Denoising model to use [n2v/n2n]. Defualt: n2v')
parser.add_argument('--data_directory', '-d', default=os.getcwd(), help='Directory where training files live.')
args = parser.parse_args()

batchsize = 1
num_steps = 50 // batchsize

# Selecting model
if args.denoising_model == 'n2v':   # Noise2Void
    model      = UNetN2V(in_ch=1, depth=3).to(DEVICE)
    criterion  = nn.MSELoss(reduction='none')
    model_name = 'n2v.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0004)

if args.denoising_model == 'n2n':   # Noise2Noise
    model      = UNetN2N(in_ch=1, depth=5).to(DEVICE)
    criterion  = nn.MSELoss()
    model_name = 'n2n.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.0001)


# Grabbing files
directory = Path(args.data_directory)
file_list = sorted(directory.glob('*.fits'))

if args.denoising_model == 'n2v':
    for i, file in enumerate(file_list[:8]):
        science_data = fits.getdata(file).astype(float)  
        
        train_loader = torch.from_numpy(science_data).float().to(DEVICE)
        train_loader = train_loader.unsqueeze(0).unsqueeze(0)
        # breakpoint()
        for ii in range(num_steps):
            B, C, H, W = train_loader.shape
            position   = []
            patch_size = 128

            for _ in range(batchsize):
                x = torch.randint(0, H - patch_size + 1, (1,))      # - patchsize ensures the patch won't be outside image limits
                y = torch.randint(0, W - patch_size + 1, (1, ))
                position.append((x, y))

            input_sequence     = random_crop(train_loader, position, batchsize, patch_size)   # crops random 128x128 from image  
            input_masked, mask = apply_n2v_mask(input_sequence)         # masks ~1% of pixels to be trained on (only N2V)

            output_seq = model(input_masked)

            loss_map = criterion(output_seq, input_sequence)
            loss     = loss_map[mask].mean()                           # only care about loss of masked values
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ii % 10 == 0:
                print(f"Epoch: {i} | Batch: {ii} | Loss: {loss}")
                plot_sample(input_sequence, output_seq)

        del file, train_loader
        gc.collect()


elif args.denoising_model == 'n2n':
    for i in range(0, len(file_list) - 1):
        file1, file2  = file_list[i], file_list[i + 1]

        science_data1 = fits.getdata(file1).astype(float)
        train_loader  = torch.from_numpy(science_data1).float().to(DEVICE)
        train_loader  = train_loader.unsqueeze(0).unsqueeze(0)

        science_data2 = fits.getdata(file2).astype(float)
        target_loader = torch.from_numpy(science_data2).float().to(DEVICE)
        target_loader = target_loader.unsqueeze(0).unsqueeze(0)             # target data will be adjacent file
        
        for ii in range(num_steps):

            B, C, H, W = train_loader.shape
            position   = []
            patch_size = 128

            for _ in range(batchsize):
                x = torch.randint(0, H - patch_size + 1, (1,))      # - patchsize ensures the patch won't be outside image limits
                y = torch.randint(0, W - patch_size + 1, (1, ))
                position.append((x, y))
            
            input_sequence  = random_crop(train_loader, position, batchsize, patch_size)   # crops random 128x128 from train image
            target_sequence = random_crop(target_loader, position, batchsize, patch_size)  # crops same region from target image

            output_seq = model(input_sequence)
            
            loss = criterion(output_seq, target_sequence)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ii % 10 == 0:
                print(f"Epoch: {i} | Batch: {ii} | Loss: {loss}")
                plot_sample(input_sequence, output_seq)

        del file1, file2, train_loader
        gc.collect()

torch.save(model, model_name)

del model, criterion, optimizer
torch.cuda.empty_cache()