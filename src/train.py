import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from preprocess import load_text, build_vocab
from dataset import SkipGramDataset
from model import SkipGramModel


def train():
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    EMBEDDING_DIM = 100
    WINDOW_SIZE = 2
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.05

    # -----------------------------
    # Load and preprocess data
    # -----------------------------
    words = load_text("data/HP1.txt", max_words=50000)
    word_to_idx, idx_to_word = build_vocab(words)

    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    dataset = SkipGramDataset(words, word_to_idx, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # -----------------------------
    # Model, loss, optimizer
    # -----------------------------
    model = SkipGramModel(vocab_size, EMBEDDING_DIM)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(EPOCHS):
        total_loss = 0

        for center, context in dataloader:
            optimizer.zero_grad()

            logits = model(center)
            loss = criterion(logits, context)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # Save embeddings
    torch.save(model.embedding.weight.data, "embeddings.pt")
    print("Training complete. Embeddings saved.")


if __name__ == "__main__":
    train()
