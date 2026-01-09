import torch
import torch.nn as nn


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # Embedding layer: maps word index -> embedding vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Output layer: maps embedding -> vocabulary logits
        self.output = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights (important!)
        self._init_weights()

    def _init_weights(self):
        # Small Gaussian initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, center_word_idx):
        """
        center_word_idx: Tensor of shape (batch_size,)
        """
        # (batch_size, embedding_dim)
        embed = self.embedding(center_word_idx)

        # (batch_size, vocab_size)
        logits = self.output(embed)

        return logits
