from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils import config

path = Path('data')
train_sequences = pd.read_csv(path / "train_sequences.csv")  # (844, 5)
train_labels = pd.read_csv(path / "train_labels.csv")  # (137095, 7)
train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0] + '_' + x.split("_")[1])

cache_file = path / "processed_data.pkl"

print("Processing dataset...")

all_xyz = []
for pdb_id in tqdm(train_sequences['target_id']):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[['x_1', 'y_1', 'z_1']].to_numpy().astype('float32')

    xyz[xyz < -1e17] = float('Nan')  # Handle invalid values
    all_xyz.append(xyz)

# **Log original scale**
xyz_all = np.vstack([xyz for xyz in all_xyz if len(xyz) > 0])
mean_xyz = np.nanmean(xyz_all, axis=0)
std_xyz = np.nanstd(xyz_all, axis=0)

print(f"Before Normalization:\nMean: {mean_xyz}, Std: {std_xyz}")

# **Normalize XYZ**
all_xyz = [(xyz - mean_xyz) / (std_xyz + 1e-6) for xyz in all_xyz]

# **Log normalized scale**
xyz_all_norm = np.vstack([xyz for xyz in all_xyz if len(xyz) > 0])
mean_xyz_norm = np.nanmean(xyz_all_norm, axis=0)
std_xyz_norm = np.nanstd(xyz_all_norm, axis=0)

print(f"After Normalization:\nMean: {mean_xyz_norm}, Std: {std_xyz_norm}")

# Filter sequences
filter_nan = []
max_len = 0
for xyz in all_xyz:
    if len(xyz) > max_len:
        max_len = len(xyz)
    filter_nan.append((np.isnan(xyz).mean() <= 0.5) &
                      (len(xyz) < config['max_len_filter']) &
                      (len(xyz) > config['min_len_filter']))

print(f"Longest sequence in train: {max_len}")

filter_nan = np.array(filter_nan)
non_nan_indices = np.arange(len(filter_nan))[filter_nan]

train_sequences = train_sequences.loc[non_nan_indices].reset_index(drop=True)
all_xyz = [all_xyz[i] for i in non_nan_indices]

data = {
    "sequence": train_sequences['sequence'].to_list(),
    "temporal_cutoff": train_sequences['temporal_cutoff'].to_list(),
    "description": train_sequences['description'].to_list(),
    "all_sequences": train_sequences['all_sequences'].to_list(),
    "xyz": all_xyz
}

with open(cache_file, "wb") as f:
    pickle.dump(data, f)

print("Preprocessed data saved.")

