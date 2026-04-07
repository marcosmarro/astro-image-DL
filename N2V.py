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


def apply_n2v_mask(patches, mask_ratio=0.02):
    """
    Apply N2V masking: replace masked pixels with a random neighbour value.
    Returns (masked_input, mask).
    """
    B, C, H, W = patches.shape
    input_seq = patches.clone()
    mask = torch.rand(patches.shape) < mask_ratio

    for b in range(B):
        for c in range(C):
            coords = mask[b, c].nonzero(as_tuple=False)
            for (py, px) in coords:
                dy = torch.randint(-2, 3, (1,)).item()
                dx = torch.randint(-2, 3, (1,)).item()
                ny = int(py) + dy
                nx = int(px) + dx
                ny = max(0, min(ny, H - 1))
                nx = max(0, min(nx, W - 1))
                input_seq[b, c, py, px] = patches[b, c, ny, nx]

    return input_seq, mask


def extract_patch(batch, patch_size):
    """Extract one random patch per image in the batch."""
    B, C, H, W = batch.shape
    patches = torch.zeros((B, C, patch_size, patch_size), device=batch.device)
    for i in range(B):
        x = torch.randint(0, H - patch_size + 1, (1,)).item()
        y = torch.randint(0, W - patch_size + 1, (1,)).item()
        patches[i] = batch[i, :, x:x + patch_size, y:y + patch_size]
    return patches


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_name  = args.data[:-4]
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    model      = UNetN2V(in_ch=1, depth=3).to(DEVICE)
    criterion  = nn.MSELoss(reduction='none')
    model_name = 'n2v.pth'
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(int(num_epochs/5), num_epochs, step=int(num_epochs/5)), gamma=0.5)

    # Best model tracking / early stopping
    best_model = model.state_dict()
    best_loss  = float('inf')
    best_epoch = 0
    patience   = 5
    counter    = 0

    train_dataset = LPSEB_Dataset(args.data, type='train')
    val_dataset   = LPSEB_Dataset(args.data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False)

    for epoch in range(1, num_epochs + 1):

        # Training
        model.train()
        train_count, train_loss = 0, 0.0

        for batch in tqdm(train_loader, leave=False):
            batch   = batch.to(DEVICE)
            patches = extract_patch(batch, patch_size)                  # (B,C,P,P) on DEVICE

            input_seq, mask = apply_n2v_mask(patches)                  # neighbour-replace masking
            mask            = mask.to(DEVICE)
            input_seq       = input_seq.to(DEVICE)

            output_seq = model(input_seq)

            loss_map = criterion(output_seq, patches)
            loss     = loss_map[mask].mean()                            # loss only on masked pixels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_count += 1
            train_loss  += loss.item()

        plot_sample(patches, model(input_seq))

        # Validation
        model.eval()
        val_count, val_loss = 0, 0.0

        with torch.inference_mode():
            for batch in tqdm(val_loader, leave=False):
                batch   = batch.to(DEVICE)
                patches = extract_patch(batch, patch_size)

                input_seq, mask = apply_n2v_mask(patches)
                mask            = mask.to(DEVICE)
                input_seq       = input_seq.to(DEVICE)

                output_seq = model(input_seq)

                loss_map = criterion(output_seq, patches)
                val_loss += loss_map[mask].mean().item()
                val_count += 1

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

    print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_loss:.4f}")
    model.load_state_dict(best_model)

    if args.save_model:
        torch.save(model.state_dict(), model_name)
        print(f"Model saved as {model_name}")

    model.eval()

    # Infer output shape from first batch to avoid H/W scope issues
    sample_batch = next(iter(val_loader)).to(DEVICE)
    _, _, H, W   = sample_batch.shape

    denoised = np.zeros((len(val_dataset), H, W))

    with torch.inference_mode():
        for i, batch in tqdm(enumerate(val_loader)):
            batch = batch.to(DEVICE)

            output_sequence = model(batch)

            denoised[i] = output_sequence.cpu().numpy()

    np.save(f'{file_name}_N2V.npy', denoised)
    print(f'Denoised images saved as {file_name}_N2V.npy')

    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet for denoising astronomical images using Noise2Void.")
    parser.add_argument('--data',       type=str,   default=None,  help='Path to the .npy file containing registered astronomy frames.')
    parser.add_argument('--batch_size', type=int,   default=2,     help='Batch size for training.')
    parser.add_argument('--patch_size', type=int,   default=128,   help='Patch size for training.')
    parser.add_argument('--num_epochs', type=int,   default=25,    help='Number of training epochs.')
    parser.add_argument('--lr',         type=float, default=5e-4,  help='Learning rate for the optimizer.')
    parser.add_argument('--save_model', type=bool,  default=True, help='Whether to save the trained model.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)