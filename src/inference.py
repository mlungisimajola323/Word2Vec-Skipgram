import torch


def get_embedding(word, word_to_idx, embeddings):
    """
    Returns the embedding vector for a given word.
    """
    idx = word_to_idx[word]
    return embeddings[idx]


if __name__ == "__main__":
    embeddings = torch.load("embeddings.pt")
    print("Embedding shape:", embeddings.shape)
