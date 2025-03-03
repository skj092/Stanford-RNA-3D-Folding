from utils import generate_predictions, visualize_3d_structure, create_submission_file
from pathlib import Path
from utils import load_pickle, extract_sequence_features, enhance_features_with_ss
from dataset import RNADataset, collate_fn
import pandas as pd
from torch.utils.data import DataLoader
import torch
from model import RNAFoldingModel

# 13. Main Execution


def main():
    # 1. Data Loading and Exploration
    print("Loading datasets...")
    path = Path('./data')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "model.pth"

    test_sequences = pd.read_csv(path/'test_sequences.csv')
    sample_submission = pd.read_csv(path/'sample_submission.csv')

    # subset preprocessed file
    test_preprocessed_data = Path("preprocessed_test.pkl")

    print(f"Test sequences: {test_sequences.shape}")
    print(f"Sample submission: {sample_submission.shape}")

    print("Preprocessing training data...")
    test_data = load_pickle(test_preprocessed_data)

    for i, data in enumerate(test_data):
        test_data[i]['features'] = extract_sequence_features(data['sequence'])
    print("Enhancing features with secondary structure information...")
    test_data = enhance_features_with_ss(test_data)

    # Dataset and DataLoader
    test_dataset = RNADataset(test_data)

    batch_size = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn)

    print("\n--- Main execution ---")
    print(f"Using device: {device}")

    input_dim = test_data[0]['features'].shape[1]
    trained_model = RNAFoldingModel(input_dim=input_dim).to(device)
    trained_model.load_state_dict(torch.load(model_path, weights_only=True))
    trained_model.eval()
    print("\nModel loaded.")

    print("\nStarting model training...")
    print("\nModel training finished.")

    print("\nGenerating predictions on test data...")
    test_predictions = generate_predictions(
        trained_model, test_loader, device, num_predictions=5)
    print("\nPredictions generated.")

    print("\nCreating submission file...")
    submission_file = create_submission_file(test_predictions, test_sequences)
    print(f"\nSubmission file created: {submission_file}")
    print(submission_file.head())

    print("\nVisualizing a sample prediction (first test sequence)...")
    sample_seq_id = test_sequences['target_id'].iloc[0]
    if sample_seq_id in test_predictions:
        sample_prediction = test_predictions[sample_seq_id][0]
        visualize_3d_structure(
            sample_prediction, title=f"Predicted 3D Structure - {sample_seq_id}")
        print(f"Visualization saved for {sample_seq_id}.")
    else:
        print("No prediction found for the first test sequence for visualization.")

    print("\n--- Main execution completed ---")


if __name__ == '__main__':
    main()

print("\nNotebook execution finished.")
