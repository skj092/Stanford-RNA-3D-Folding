from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F


class RNA3D_Dataset(Dataset):
    def __init__(self, indices, data, config):
        self.config = config
        self.indices = indices
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}
        # Default to 384 if not specified
        self.max_len = config.get('max_len', 384)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = self.indices[index]

        # Encode sequence
        sequence = [self.tokens[nt] for nt in self.data['sequence'][idx]]
        sequence = torch.tensor(sequence, dtype=torch.long)

        # Load xyz coordinates
        xyz = torch.tensor(self.data['xyz'][idx], dtype=torch.float32)

        # Handle cropping/padding
        seq_len = len(sequence)

        if seq_len > self.max_len:
            # Crop sequence randomly
            crop_start = np.random.randint(seq_len - self.max_len)
            crop_end = crop_start + self.max_len
            sequence = sequence[crop_start:crop_end]
            xyz = xyz[crop_start:crop_end]
        elif seq_len < self.max_len:
            # Pad sequence with 0s
            pad_size = self.max_len - seq_len
            sequence = F.pad(sequence, (0, pad_size),
                             value=0)  # Padding with 0
            # Padding (N, 3) with 0
            xyz = F.pad(xyz, (0, 0, 0, pad_size), value=0)

        return sequence, xyz
