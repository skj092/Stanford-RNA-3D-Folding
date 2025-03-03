# Stanford RNA 3D Folding Competition
# Revised Notebook for RNA 3D Structure Prediction (Improved V3)

from model import RNAFoldingModel
from dataset import RNADataset, collate_fn
from utils import extract_sequence_features, enhance_features_with_ss
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from pathlib import Path
from engine import (train_epoch, validate,
                    evaluate_model)
from utils import load_pickle
warnings.filterwarnings('ignore')


def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.0005, device='cpu'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5)

    train_losses = []
    val_losses = []
    tm_scores = []

    best_model_state = model.state_dict().copy()
    best_val_loss = float('inf')
    best_tm_score = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        tm_score = evaluate_model(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        tm_scores.append(tm_score)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  TM-Score: {tm_score:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f'  New best model saved (Val Loss: {val_loss:.4f})')
        if tm_score > best_tm_score:
            best_tm_score = tm_score
            print(f'  New best TM-score: {tm_score:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Best model state not found; using current parameters.")

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(tm_scores, label='TM-Score')
    # plt.title('TM-Score Evolution')
    # plt.xlabel('Epoch')
    # plt.ylabel('TM-Score')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('training_history.png')
    # plt.close()

    print(
        f"Training complete. Best Val Loss: {best_val_loss:.4f}, Best TM-Score: {best_tm_score:.4f}")
    return model


# 13. Main Execution
def main():
    # 1. Data Loading and Exploration
    print("Loading datasets...")
    path = Path('data')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # subset preprocessed file
    train_preprocessed_data = path/"preprocessed_train.pkl"
    valid_preprocessed_data = path/"preprocessed_valid.pkl"

    print("Preprocessing training data...")
    train_data = load_pickle(train_preprocessed_data)[:50]
    validation_data = load_pickle(valid_preprocessed_data)
    breakpoint()

    for i, data in enumerate(train_data):
        train_data[i]['features'] = extract_sequence_features(data['sequence'])
    for i, data in enumerate(validation_data):
        validation_data[i]['features'] = extract_sequence_features(
            data['sequence'])
    print("Enhancing features with secondary structure information...")
    train_data = enhance_features_with_ss(train_data)
    validation_data = enhance_features_with_ss(validation_data)

    # Dataset and DataLoader
    train_dataset = RNADataset(train_data)
    validation_dataset = RNADataset(validation_data)

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    input_dim = train_data[0]['features'].shape[1]
    model = RNAFoldingModel(input_dim=input_dim).to(device)
    print("\nModel instantiated.")

    print("\nStarting model training...")
    train_model(
        model,
        train_loader,
        validation_loader,
        num_epochs=2,  # Increased epochs
        lr=0.0005,     # Lower learning rate for stability
        device=device
    )
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()

print("\nNotebook execution finished.")
