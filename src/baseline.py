import sys
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataset import RNA3D_Dataset
from utils import get_plot, config
from torch.utils.data import DataLoader, Subset
from model import RNA3DTransformer
import torch.nn as nn
import pickle
from fastprogress.fastprogress import master_bar, progress_bar
from time import sleep


# load the traininng and label files
path = Path('data')
train_sequences = pd.read_csv(path/"train_sequences.csv")  # (844, 5)
train_labels = pd.read_csv(path/"train_labels.csv")  # (137095, 7)
train_labels["pdb_id"] = train_labels["ID"].apply(
    lambda x: x.split("_")[0]+'_'+x.split("_")[1])  # (136095, 8) where unique(pdb_id) == unique(train_sequences[target_id])

# Load preprocessed data
preprocess_data_path = path / "processed_data.pkl"
with open(preprocess_data_path, "rb") as f:
    data = pickle.load(f)

# Split data into train and test
all_index = np.arange(len(data['sequence']))
cutoff_date = pd.Timestamp(config['cutoff_date'])
test_cutoff_date = pd.Timestamp(config['test_cutoff_date'])
train_index = [i for i, d in enumerate(
    data['temporal_cutoff']) if pd.Timestamp(d) <= cutoff_date]
test_index = [i for i, d in enumerate(data['temporal_cutoff']) if pd.Timestamp(
    d) > cutoff_date and pd.Timestamp(d) <= test_cutoff_date]
print(f"Train size: {len(train_index)}")
print(f"Test size: {len(test_index)}")

train_dataset = RNA3D_Dataset(train_index, data, config)
valid_dataset = RNA3D_Dataset(test_index, data, config)

# Example: Generate an Nx3 matrix
# xyz = train_dataset[200]['xyz']  # Replace this with your actual Nx3 data
# N = len(xyz)
# get_plot(xyz)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

train_loader = Subset(train_dataset, range(100))

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNA3DTransformer()
model.to(device)
print(model)

# training configs
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["learning_rate"])

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["learning_rate"])

# validate on one batch of data
xb, yb = next(iter(train_loader))
out = model(xb)
loss = criterion(yb, out)
print('one batch loss', loss)
# sys.exit()


for idx, (seqs, xyz) in enumerate(train_loader):
    seqs, xyz = seqs.to(device), xyz.to(device)
    print(f"batch index {idx}")
    print(seqs.shape, xyz.shape)
    print(xyz)

    # optimizer.zero_grad()
    # outputs = model(seqs)
    # loss = criterion(outputs, xyz)
    # loss.backward()
    # print('loss', loss)
    # optimizer.step()

# num_epochs = config["epochs"]
#
# mb = master_bar(range(10))
# for i in mb:
#     model.train()
#     train_loss = 0
#     for seqs, xyz in progress_bar(train_loader, parent=mb):
#         sleep(0.01)
#         seqs, xyz = seqs.to(device), xyz.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(seqs)
#         loss = criterion(outputs, xyz)
#         mb.child.comment = f'{loss.item():.4f}'
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#     mb.write(
#         f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}")
