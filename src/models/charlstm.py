import torch
from torch import nn

class CharLSTMClient(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, tokens):
        embedded = self.embedding(tokens.long())
        sequence, _ = self.lstm(embedded)
        return sequence


class CharLSTMServer(nn.Module):
    def __init__(self, hidden_size, num_layers, vocab_size):
        super().__init__()
        server_layers = max(1, num_layers - 1)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=server_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, smashed):
        sequence, _ = self.lstm(smashed)
        return self.classifier(sequence[:, -1, :])
