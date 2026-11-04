"""Complete character-level LSTM model for Federated Learning on Shakespeare."""

import torch
from torch import nn


class CharLSTM(nn.Module):
    """Complete character-level LSTM next-character classifier."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=max(1, num_layers),
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens.long())
        sequence, _ = self.lstm(embedded)
        return self.classifier(sequence[:, -1, :])


def build_charlstm(config) -> CharLSTM:
    return CharLSTM(
        vocab_size=config.num_classes,
        embed_dim=config.charlstm_embed_dim,
        hidden_size=config.charlstm_hidden_size,
        num_layers=config.charlstm_num_layers,
    )

