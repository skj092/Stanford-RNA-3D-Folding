from pathlib import Path
import pandas as pd
from utils import preprocess_sequence_data, save_pickle, load_pickle


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

    # n_sample = 50
    # train_sequences = train_sequences.sample(n_sample)
    # train_labels = train_labels.sample(n_sample)
    # Fill missing coordinate values to avoid NaNs.
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

    if train_preprocessed_data.exists():
        print("Loading cached training data...")
        train_data = load_pickle(train_preprocessed_data)
    else:
        print("Processing and caching training data...")
        train_data = preprocess_sequence_data(train_sequences, train_labels)
        save_pickle(train_data, train_preprocessed_data)

    if valid_preprocessed_data.exists():
        print("Loading cached validation data...")
        validation_data = load_pickle(valid_preprocessed_data)
    else:
        print("Processing and caching validation data...")
        validation_data = preprocess_sequence_data(
            validation_sequences, validation_labels)
        save_pickle(validation_data, valid_preprocessed_data)

    if test_preprocessed_data.exists():
        print("Loading cached test data...")
        test_data = load_pickle(test_preprocessed_data)
    else:
        print("Processing and caching test data...")
        test_data = preprocess_sequence_data(test_sequences, is_train=False)
        save_pickle(test_data, test_preprocessed_data)
