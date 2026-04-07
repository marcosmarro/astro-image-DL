import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from network import UNetN2N
from torch.utils.data import DataLoader
# Custom imports
from data import LPSEB_Dataset_N2N
from utils import plot_sample
torch.manual_seed(42)


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_name  = args.data[:-4]
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    dpth = 4
    # Selecting model
    model      = UNetN2N(in_ch=1, depth=dpth).to(DEVICE)
    criterion  = nn.MSELoss()
    model_name = f'n2n_{dpth}.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(int(num_epochs/5), num_epochs, step=int(num_epochs/5)), gamma=0.5)

    # Setting up best model saving and early stopping
    best_model = model.state_dict()
    best_loss  = float('inf')
    best_epoch = 0
    patience   = 5
    counter    = 0

    # Grabbing files
    all_files   = LPSEB_Dataset_N2N(args.data)

    train_dataset = LPSEB_Dataset_N2N(args.data, type='train')
    val_dataset   = LPSEB_Dataset_N2N(args.data, type='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)
    all_loader   = DataLoader(all_files, batch_size=1, shuffle=False)

    for epoch in range(1, num_epochs + 1):

        # Training
        model.train()
        train_count, train_loss = 0, 0.0

        for batch in tqdm(train_loader, leave=False):
            batch = batch.to(DEVICE)
            
            B, C, N, H, W = batch.shape

            # Allocate on the correct device and sample independent (x,y) per batch item
            x = torch.randint(0, H - patch_size + 1, (B,))
            y = torch.randint(0, W - patch_size + 1, (B,))

            frame1 = torch.stack([batch[i, :, 0, x[i]:x[i]+patch_size, y[i]:y[i]+patch_size] for i in range(B)])
            frame2 = torch.stack([batch[i, :, 1, x[i]:x[i]+patch_size, y[i]:y[i]+patch_size] for i in range(B)])

            output_seq = model(frame1)

            loss = criterion(output_seq, frame2)

            train_count += 1
            train_loss  += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plot_sample(frame1, output_seq)

        # Validation
        val_count, val_loss = 0, 0.0
        model.eval()
        with torch.inference_mode():
            for batch in tqdm(test_loader, leave=False):
                batch = batch.to(DEVICE)

                frame1 = batch[:, :, 0, :, :]
                frame2 = batch[:, :, 1, :, :]

                output_seq = model(frame1)

                val_count += 1
                val_loss  += ((output_seq - frame2) ** 2).mean().item()

        avg_train = train_loss / train_count
        avg_val   = val_loss   / val_count

        print(f"Epoch: {epoch:3d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.3e}")

        # Early stopping
        if avg_val < best_loss:
            best_loss  = avg_val
            best_model = model.state_dict()
            best_epoch = epoch
            counter    = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        scheduler.step()

    print(f"Best Epoch: {best_epoch} | Best Loss: {best_loss:.4f}")
    model.load_state_dict(best_model)

    if args.save_model:
        torch.save(model.state_dict(), model_name)
        print(f"Model saved as {model_name}")

    model.eval()

    denoised = np.zeros((len(all_files), H, W))

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(all_loader)):
            batch = batch[:, :, 0, :, :].to(DEVICE)

            output_sequence = model(batch)

            denoised[i] = output_sequence.cpu().numpy()

    np.save(f'{file_name}_N2N.npy', denoised)
    print(f'Denoised images saved as {file_name}_N2N.npy')

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet for denoising astronomical images using Noise2Noise.")
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