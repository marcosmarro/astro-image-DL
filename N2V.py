import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from network import UNetN2V
from torch.utils.data import DataLoader
# Custom imports
from data import LPSEB_Dataset
from utils import plot_sample
torch.manual_seed(42)


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_name  = args.data[:-4]
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # Selecting model
    model      = UNetN2V(in_ch=1, depth=3).to(DEVICE)
    criterion  = nn.MSELoss(reduction='none')
    model_name = 'n2v.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(num_epochs, step=int(num_epochs/5)), gamma=0.5)

    # Setting up best model saving and early stopping
    best_model = model.state_dict()
    best_loss  = float('inf')
    best_epoch = 0
    patience   = 5
    counter    = 0

    # Grabbing files
    train, test  = LPSEB_Dataset(args.data, type='train'), LPSEB_Dataset(args.data)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test, batch_size=1, shuffle=False)
    for epoch in range(1, num_epochs + 1):

        train_count, train_loss = 0, 0.0

        for batch in tqdm(train_loader, leave=False):
            batch = batch.to(DEVICE)
            B, C, H, W = batch.shape
            breakpoint()

            patches = torch.zeros((B, C, patch_size, patch_size))

            for i in range(len(batch)):
                x = torch.randint(0, H - patch_size + 1, (1,))      # - patchsize ensures the patch won't be outside image limits
                y = torch.randint(0, W - patch_size + 1, (1, ))
                patches[i:i+1] = batch[i:i+1, :, x:x+patch_size, y:y+patch_size]

            input_seq = patches.clone()
            mask      = torch.rand((patches.shape)) < 0.01

            input_seq[mask] = 0.0 
            
            output_seq = model(input_seq)

            loss_map = criterion(output_seq, patches)
            loss     = loss_map[mask].mean()                        # only care about loss of masked values

            train_count += 1
            train_loss  += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plot_sample(patches, output_seq)

        test_count, test_loss = 0, 0.0

        with torch.inference_mode():
            for batch in tqdm(test_loader, leave=False):
                batch = batch.to(DEVICE)
                B, C, H, W = batch.shape

                input_seq = torch.zeros((B, C, patch_size, patch_size))
                
                for i in range(len(batch)):
                    x = torch.randint(0, H - patch_size + 1, (1,))      # - patchsize ensures the patch won't be outside image limits
                    y = torch.randint(0, W - patch_size + 1, (1, ))
                    input_seq[i:i+1] = batch[i:i+1, :, x:x+patch_size, y:y+patch_size]
                
                output_seq = model(input_seq)

                test_count += 1
                test_loss  += ((output_seq - input_seq) ** 2).mean().item()

        print(f"Epoch: {epoch} | Train Loss: {train_loss / train_count:.4f} | Test Loss: {test_loss / test_count:.4f} | LR: {scheduler.get_last_lr()[0]:.3e}")

        if test_loss / test_count < best_loss:
            best_loss  = test_loss / test_count
            best_model = model.state_dict()
            best_epoch = epoch
            counter    = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

        scheduler.step()

    print(f"Best Epoch: {best_epoch} | Best Loss: {best_loss:.4f}")
    model.load_state_dict(best_model)

    if args.save_model:
        torch.save(model, model_name)

    denoised = np.zeros((len(test), H, W))

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(test_loader)):
            batch = batch.to(DEVICE)

            output_sequence = model(batch)

            denoised[i] = output_sequence.cpu().numpy()

    np.save(f'{file_name}_N2V.npy', denoised)
    print(f'Denoised images saved as {file_name}_N2V.npy')

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet for denoising astronomical images using Noise2Void or Noise2Noise.")
    parser.add_argument('--data',       type=str,   default=None, help='Path to the file containing registered astronomy files for training.')
    parser.add_argument('--batch_size', type=int,   default=2,    help='Batch size for training.')
    parser.add_argument('--patch_size', type=int,   default=128,  help='Patch size for training.')
    parser.add_argument('--num_epochs', type=int,   default=25,   help='Number of training epochs.')
    parser.add_argument('--lr',         type=float, default=5e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--save_model', type=bool,  default=True, help='Whether to save the trained model.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)