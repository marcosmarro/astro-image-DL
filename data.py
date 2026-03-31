### Make a dataset class for loading FITS files and applying transformations
import torch
import numpy as np
from torch.utils.data import Dataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class LPSEB_Dataset(Dataset):
    def __init__(self, data_path, type=None):
        self.data_path = data_path
        self.data = np.load(data_path).astype(np.float32)
        self.type = type
        if self.type == 'train':
            self.data = self.data[:int(0.8 * len(self.data))]  # Use first 80% of frames for training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file = self.data[idx]
        data = torch.from_numpy(file).unsqueeze(0)  # add channel dimension

        return data
    

class LPSEB_Dataset_N2N(Dataset):
    def __init__(self, data_path, type=None):
        self.data_path = data_path
        self.data = np.load(data_path).astype(np.float32)  # (N, H, W)
        self.type = type
        if self.type == 'train':
            self.data = self.data[:int(0.9 * len(self.data))]  # Use first 80% of frames for training
        if self.type == 'test':
            self.data = self.data[int(0.9 * len(self.data)):]  # Use last 20% of frames for testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Pick a second frame randomly, ensuring it's different from idx
        idx2 = np.random.randint(0, len(self.data))
        while idx2 == idx:
            idx2 = np.random.randint(0, len(self.data))

        frame1 = torch.from_numpy(self.data[idx])    # (H, W)
        frame2 = torch.from_numpy(self.data[idx2])   # (H, W)

        pair = torch.stack([frame1, frame2], dim=0).unsqueeze(0)  # (C, 2, H, W)

        return pair