from pathlib import Path
import pandas as pd
from utils import save_pickle
from tqdm import tqdm
import numpy as np


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


if __name__ == "__main__":
    # 1. Data Loading and Exploration
    print("Loading datasets...")
    path = Path('data')
    train_sequences = pd.read_csv(path/'train_sequences.csv')
    train_labels = pd.read_csv(path/'train_labels.csv')
    validation_sequences = pd.read_csv(path/'validation_sequences.csv')
    validation_labels = pd.read_csv(path/'validation_labels.csv')
    test_sequences = pd.read_csv(path/'test_sequences.csv')
    sample_submission = pd.read_csv(path/'sample_submission.csv')

    train_labels.fillna(0, inplace=True)
    validation_labels.fillna(0, inplace=True)

    print("\nBasic dataset information:")
    print(f"Training sequences: {train_sequences.shape}")
    print(f"Training labels: {train_labels.shape}")
    print(f"Validation sequences: {validation_sequences.shape}")
    print(f"Validation labels: {validation_labels.shape}")
    print(f"Test sequences: {test_sequences.shape}")
    print(f"Sample submission: {sample_submission.shape}")

    print("Preprocessing training data...")
    train_preprocessed_data = path/"preprocessed_train.pkl"
    valid_preprocessed_data = path/"preprocessed_valid.pkl"
    test_preprocessed_data = path/"preprocessed_test.pkl"

    train_data = preprocess_sequence_data(train_sequences, train_labels)
    save_pickle(train_data, train_preprocessed_data)

    validation_data = preprocess_sequence_data(
        validation_sequences, validation_labels)
    save_pickle(validation_data, valid_preprocessed_data)

    test_data = preprocess_sequence_data(test_sequences, is_train=False)
    save_pickle(test_data, test_preprocessed_data)
