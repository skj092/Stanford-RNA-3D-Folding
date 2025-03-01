import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig


class RNA3DTransformer(nn.Module):
    def __init__(self, vocab_size=5, hidden_dim=256, num_layers=4, num_heads=8):
        super(RNA3DTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        # Predict 3D coordinates
        self.regressor = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.embedding(x)  # Convert sequence tokens to embeddings
        x = self.transformer(x)  # Pass through Transformer Encoder
        xyz = self.regressor(x)  # Predict 3D coordinates
        return xyz


if __name__ == "__main__":
    # Define model
    model = RNA3DTransformer()
    print(model)
