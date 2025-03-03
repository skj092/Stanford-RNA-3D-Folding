import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import pickle


def preprocess_sequence_data(sequences_df, labels_df=None, is_train=True):
    """
    Preprocess RNA sequence data.
    Convert sequences to numerical form and normalize coordinate targets per sequence.
    """
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
    processed_data = []

    for idx, row in tqdm(sequences_df.iterrows()):
        seq_id = row['target_id']
        sequence = row['sequence']
        numerical_seq = [nucleotide_map.get(nuc, 4) for nuc in sequence]

        structures = None
        if is_train and labels_df is not None:
            sequence_labels = labels_df[labels_df['ID'].str.startswith(
                seq_id + '_')]
            if not sequence_labels.empty:
                num_structures = (len(sequence_labels.columns) - 3) // 3
                structures = []
                for i in range(1, num_structures + 1):
                    coords = []
                    for _, label_row in sequence_labels.iterrows():
                        x = label_row[f'x_{i}']
                        y = label_row[f'y_{i}']
                        z = label_row[f'z_{i}']
                        coords.append([x, y, z])
                    coords = np.array(coords)
                    # Normalize coordinates per sequence (center and scale)
                    mean = np.mean(coords, axis=0)
                    std = np.std(coords, axis=0) + 1e-8
                    coords_norm = (coords - mean) / std
                    structures.append(coords_norm)
        processed_data.append({
            'id': seq_id,
            'sequence': numerical_seq,
            'structures': structures
        })
    return processed_data


# 4. Feature Engineering
def extract_sequence_features(sequence):
    """
    Extract one-hot encoding, positional encoding, and GC-content as features.
    """
    one_hot = np.zeros((len(sequence), 5))
    for i, nucleotide in enumerate(sequence):
        one_hot[i, nucleotide] = 1
    gc_content = []
    window_size = 5
    for i in range(len(sequence)):
        start = max(0, i - window_size // 2)
        end = min(len(sequence), i + window_size // 2 + 1)
        window = sequence[start:end]
        gc_count = sum(1 for n in window if n in [1, 2])
        gc_content.append(gc_count / len(window))
    positions = np.array([[i / len(sequence)] for i in range(len(sequence))])
    features = np.hstack(
        (one_hot, positions, np.array(gc_content).reshape(-1, 1)))
    return features


print("Starting Stanford RNA 3D Folding notebook...")


# 5. RNA Secondary Structure Prediction (simple rule-based)
def predict_rna_secondary_structure(sequence):
    nucleotide_map_inv = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: 'X'}
    seq_chars = [nucleotide_map_inv[n] for n in sequence]
    structure = ['.' for _ in range(len(seq_chars))]
    complementary = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'X': None}
    for i in range(len(seq_chars)):
        if structure[i] != '.':
            continue
        for j in range(len(seq_chars) - 1, i + 3, -1):
            if structure[j] != '.':
                continue
            if complementary[seq_chars[i]] == seq_chars[j]:
                structure[i] = '('
                structure[j] = ')'
                break
    return ''.join(structure)


def enhance_features_with_ss(data):
    for i, item in enumerate(data):
        seq = item['sequence']
        ss = predict_rna_secondary_structure(seq)
        ss_features = np.zeros((len(ss), 3))
        for j, char in enumerate(ss):
            if char == '.':
                ss_features[j, 0] = 1
            elif char == '(':
                ss_features[j, 1] = 1
            elif char == ')':
                ss_features[j, 2] = 1
        data[i]['features'] = np.hstack((item['features'], ss_features))
    return data


# 9. Model Inference and Multiple Structure Generation
def generate_diverse_structures(model, features, seq_length, num_structures=5, noise_scale=0.05, device='cpu'):
    model.eval()
    structures = []
    for i in range(num_structures):
        with torch.no_grad():
            if i > 0:
                noise = torch.randn_like(features) * noise_scale
                features_with_noise = features + noise
            else:
                features_with_noise = features
            output = model(features_with_noise.unsqueeze(0))
            coords = output[0, :seq_length, :].cpu().numpy()
            structures.append(coords)
    return structures


def generate_predictions(model, dataloader, device, num_predictions=5):
    model.eval()
    all_predictions = {}
    for features, _, ids, seq_lengths in dataloader:
        features = features.to(device)
        for i, (seq_id, length) in enumerate(zip(ids, seq_lengths)):
            seq_features = features[i, :length, :]
            predictions = generate_diverse_structures(
                model,
                seq_features,
                length,
                num_structures=num_predictions,
                device=device
            )
            all_predictions[seq_id] = predictions
    return all_predictions


# 10. Submission File Generation
def create_submission_file(predictions, test_sequences_df, output_file='submission.csv'):
    submission_rows = []
    for _, row in test_sequences_df.iterrows():
        seq_id = row['target_id']
        sequence = row['sequence']
        if seq_id in predictions:
            pred_structures = predictions[seq_id]
            num_structures = len(pred_structures)
            for i in range(len(sequence)):
                submission_row = {
                    'ID': f"{seq_id}_{i+1}",
                    'resname': sequence[i],
                    'resid': i+1
                }
                for j in range(5):
                    if j < num_structures:
                        coords = pred_structures[j][i]
                        submission_row[f'x_{j+1}'] = coords[0]
                        submission_row[f'y_{j+1}'] = coords[1]
                        submission_row[f'z_{j+1}'] = coords[2]
                    else:
                        submission_row[f'x_{j+1}'] = submission_row[f'x_{j}']
                        submission_row[f'y_{j+1}'] = submission_row[f'y_{j}']
                        submission_row[f'z_{j+1}'] = submission_row[f'z_{j}']
                submission_rows.append(submission_row)
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_file, index=False)
    return submission_df


# 11. Visualization Functions
def visualize_3d_structure(coords, title="RNA 3D Structure"):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c='blue', marker='o', s=30, label="C1' atoms")
    for i in range(len(coords) - 1):
        ax.plot([coords[i, 0], coords[i+1, 0]],
                [coords[i, 1], coords[i+1, 1]],
                [coords[i, 2], coords[i+1, 2]], 'k-', lw=1)
    ax.set_title(title)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


# 12. (Optional) Ensemble Modeling
class ModelEnsemble:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [
            1/len(models)] * len(models)

    def predict(self, features, seq_lengths=None):
        all_predictions = []
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                output = model(features, seq_lengths)
                all_predictions.append(output * self.weights[i])


def save_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
