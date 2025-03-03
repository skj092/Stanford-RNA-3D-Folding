import torch.nn as nn
import torch


# 7. Model Architecture with BatchNorm and increased capacity
class RNAFoldingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super(RNAFoldingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 3)  # 3D coordinates
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, seq_lengths=None):
        batch_size, seq_len, _ = x.size()
        if seq_lengths is not None:
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths, batch_first=True, enforce_sorted=True)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        context_vector = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        combined = lstm_out + context_vector
        x = self.relu(self.fc1(combined))
        # BatchNorm expects input as (B, C, L), so transpose, apply, then transpose back
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
