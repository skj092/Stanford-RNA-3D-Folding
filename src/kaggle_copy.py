import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch for deep learning model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define file paths (Kaggle input paths)
TRAIN_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/train_sequences.csv'
TRAIN_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/train_labels.csv'
VALID_SEQ_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv'
VALID_LABELS_PATH = '/kaggle/input/stanford-rna-3d-folding/validation_labels.csv'
TEST_SEQ_PATH  = '/kaggle/input/stanford-rna-3d-folding/test_sequences.csv'
SAMPLE_SUB_PATH = '/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'

# Load CSV files
train_sequences = pd.read_csv(TRAIN_SEQ_PATH)
train_labels = pd.read_csv(TRAIN_LABELS_PATH)
valid_sequences = pd.read_csv(VALID_SEQ_PATH)
valid_labels = pd.read_csv(VALID_LABELS_PATH)
test_sequences = pd.read_csv(TEST_SEQ_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

# Fill missing values in labels with 0
train_labels.fillna(0, inplace=True)
valid_labels.fillna(0, inplace=True)

# Display basic info
print("Train Sequences Shape:", train_sequences.shape)
print("Train Labels Shape:", train_labels.shape)
print("Validation Sequences Shape:", valid_sequences.shape)
print("Validation Labels Shape:", valid_labels.shape)
print("Test Sequences Shape:", test_sequences.shape)

# Look at a few examples
print("\nTrain Sequences Head:")
print(train_sequences.head())
print("\nTrain Labels Head:")
print(train_labels.head())

"""## 3. Data Preprocessing

### 3.1 Sequence Encoding

We map each nucleotide to an integer:
- A: 1, C: 2, G: 3, U: 4
Unknown characters are mapped to 0.
"""

nucleotide_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}

def encode_sequence(seq):
    """Encodes a RNA sequence into a list of integers based on nucleotide_map."""
    return [nucleotide_map.get(ch, 0) for ch in seq]

# Apply encoding to all sequence files
train_sequences['encoded'] = train_sequences['sequence'].apply(encode_sequence)
valid_sequences['encoded'] = valid_sequences['sequence'].apply(encode_sequence)
test_sequences['encoded'] = test_sequences['sequence'].apply(encode_sequence)

"""### 3.2 Processing Label Data

Each row in the labels CSV is for one residue, with an `ID` formatted as `target_id_resid`.
We group rows by `target_id` and sort by residue number.
Here, we use the first structure (x_1, y_1, z_1) as our target coordinates.
"""

def process_labels(labels_df):
    """
    Processes a labels DataFrame by grouping rows by target_id.
    Returns a dictionary mapping target_id to an array of coordinates (seq_len, 3).
    """
    label_dict = {}
    for idx, row in labels_df.iterrows():
        # Split ID into target_id and residue number (assumes format "targetid_resid")
        parts = row['ID'].split('_')
        target_id = "_".join(parts[:-1])
        resid = int(parts[-1])
        # Extract the coordinates; they should be numeric (missing values already set to 0)
        coord = np.array([row['x_1'], row['y_1'], row['z_1']], dtype=np.float32)
        if target_id not in label_dict:
            label_dict[target_id] = []
        label_dict[target_id].append((resid, coord))

    # Sort residues by resid and stack coordinates
    for key in label_dict:
        sorted_coords = sorted(label_dict[key], key=lambda x: x[0])
        coords = np.stack([c for r, c in sorted_coords])
        label_dict[key] = coords
    return label_dict

# Process training and validation labels
train_labels_dict = process_labels(train_labels)
valid_labels_dict = process_labels(valid_labels)

"""### 3.3 Creating Datasets and Padding

We match each target sequence with its corresponding coordinate labels.
Then we pad sequences and coordinate arrays to a uniform length.

Padded positions in coordinates are set to 0.
"""

def create_dataset(sequences_df, labels_dict):
    """
    Creates a dataset from a sequences DataFrame and a labels dictionary.
    Returns:
        X: list of encoded sequences,
        y: list of coordinate arrays,
        target_ids: list of target ids.
    """
    X, y, target_ids = [], [], []
    for idx, row in sequences_df.iterrows():
        tid = row['target_id']
        if tid in labels_dict:
            X.append(row['encoded'])
            y.append(labels_dict[tid])
            target_ids.append(tid)
    return X, y, target_ids

# Create training and validation datasets
X_train, y_train, train_ids = create_dataset(train_sequences, train_labels_dict)
X_valid, y_valid, valid_ids = create_dataset(valid_sequences, valid_labels_dict)

# Determine maximum sequence length from training set
max_len = max(len(seq) for seq in X_train)
print("Maximum sequence length (train):", max_len)

# Function to pad sequences (PyTorch version)
def pad_sequences(sequences, maxlen, padding='post', value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            # Truncate
            padded_seq = seq[:maxlen]
        else:
            # Pad
            if padding == 'post':
                padded_seq = seq + [value] * (maxlen - len(seq))
            else:  # 'pre'
                padded_seq = [value] * (maxlen - len(seq)) + seq
        padded_sequences.append(padded_seq)
    return np.array(padded_sequences)

# Pad the sequences (padding value = 0)
X_train_pad = pad_sequences(X_train, maxlen=max_len, padding='post', value=0)
X_valid_pad = pad_sequences(X_valid, maxlen=max_len, padding='post', value=0)

# Function to pad coordinate arrays
def pad_coordinates(coord_array, max_len):
    L = coord_array.shape[0]
    if L < max_len:
        pad_width = ((0, max_len - L), (0, 0))
        return np.pad(coord_array, pad_width, mode='constant', constant_values=0)
    else:
        return coord_array

# Pad coordinate arrays
y_train_pad = np.array([pad_coordinates(arr, max_len) for arr in y_train])
y_valid_pad = np.array([pad_coordinates(arr, max_len) for arr in y_valid])

# Check for any NaN values in the targets
print("Any NaN in y_train_pad?", np.isnan(y_train_pad).any())
print("Any NaN in y_valid_pad?", np.isnan(y_valid_pad).any())

print("X_train_pad shape:", X_train_pad.shape)
print("y_train_pad shape:", y_train_pad.shape)

"""## 4. Custom PyTorch Dataset and DataLoader"""

class RNAFoldingDataset(Dataset):
    def __init__(self, sequences, coordinates):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.coordinates = torch.tensor(coordinates, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.coordinates[idx]

# Create DataLoaders
batch_size = 16

train_dataset = RNAFoldingDataset(X_train_pad, y_train_pad)
valid_dataset = RNAFoldingDataset(X_valid_pad, y_valid_pad)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

"""## 5. PyTorch CNN Model Definition

In this section, we build a faster CNN-based model equivalent to the Keras version.
The model uses:
- An Embedding layer
- Two Conv1D blocks (with BatchNormalization and Dropout)
- A final Conv1D layer (kernel size 1) to output 3 coordinates per residue
"""

class RNACNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, drop_rate):
        super(RNACNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # First convolutional block
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.dropout1 = nn.Dropout(drop_rate)

        # Second convolutional block
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.dropout2 = nn.Dropout(drop_rate)

        # Final convolution to output 3 coordinates per residue
        self.conv_out = nn.Conv1d(num_filters, 3, 1, padding=0)

    def forward(self, x):
        # Shape after embedding: [batch_size, seq_len, embedding_dim]
        x = self.embedding(x)

        # Transpose for Conv1d layers which expect [batch_size, channels, seq_len]
        x = x.transpose(1, 2)

        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)

        # Final convolution
        x = self.conv_out(x)

        # Transpose back to [batch_size, seq_len, 3]
        x = x.transpose(1, 2)

        return x

# Define hyperparameters for the CNN model
vocab_size = max(nucleotide_map.values()) + 1  # +1 for padding token 0
embedding_dim = 16
num_filters = 64
kernel_size = 3
drop_rate = 0.2

# Initialize the model
model = RNACNN(vocab_size, embedding_dim, num_filters, kernel_size, drop_rate)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Display model architecture
print(model)

"""## 6. Model Training with Early Stopping

We train the PyTorch CNN model using early stopping to monitor the validation loss.
"""

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        return self.should_stop

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=50):
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training phase
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(valid_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

        # Check early stopping
        if early_stopping(epoch_val_loss, model):
            print("Early stopping triggered!")
            # Load best model
            model.load_state_dict(early_stopping.best_model_state)
            break

    # If we didn't trigger early stopping, load the best model
    if not early_stopping.should_stop:
        model.load_state_dict(early_stopping.best_model_state)

    return model, train_losses, val_losses

# Train the model
model, train_losses, val_losses = train_model(model, train_loader, valid_loader, criterion, optimizer)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss (CNN)')
plt.plot(val_losses, label='Val Loss (CNN)')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("CNN Model Training vs. Validation Loss")
plt.legend()
plt.show()

"""## 7. Generating Predictions and Submission File

For each test sequence, we predict the 3D coordinates using our trained PyTorch model.

The submission requires 5 sets of coordinates per target. In this baseline, we replicate the same predicted structure 5 times.
"""

# Prepare test data: pad sequences to same length as training
X_test = test_sequences['encoded'].tolist()
X_test_pad = pad_sequences(X_test, maxlen=max_len, padding='post', value=0)

# Convert to torch tensor
X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()

# Build submission rows. Each row corresponds to a residue from a test target.
submission_rows = []
for idx, row in test_sequences.iterrows():
    target_id = row['target_id']
    # Get predicted coordinates (shape: [max_len, 3])
    pred_coords = predictions[idx]
    # Determine actual sequence length
    seq_length = len(row['encoded'])
    pred_coords = pred_coords[:seq_length, :]  # only actual residues

    # For each residue, create a row in the submission file
    for i in range(seq_length):
        coords = pred_coords[i, :]
        # Replicate the same prediction 5 times for submission format
        submission_rows.append({
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1,
            **{f"x_{j+1}": coords[0] for j in range(5)},
            **{f"y_{j+1}": coords[1] for j in range(5)},
            **{f"z_{j+1}": coords[2] for j in range(5)}
        })

submission_df = pd.DataFrame(submission_rows)
print("Submission DataFrame shape:", submission_df.shape)
print(submission_df.head(10))

"""## 8. Saving the Submission File

Finally, we save the submission file as `submission.csv`.
"""

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
