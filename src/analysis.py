import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_text, build_vocab

# -----------------------------
# Helper functions
# -----------------------------
def nearest_neighbors(embeddings, word_to_idx, idx_to_word, target_words, top_k=5):
    """Print top_k nearest neighbors for each target word."""
    sim_matrix = cosine_similarity(embeddings)
    for word in target_words:
        if word not in word_to_idx:
            continue
        idx = word_to_idx[word]
        similarities = sim_matrix[idx]
        # Exclude the word itself
        neighbors_idx = similarities.argsort()[::-1][1:top_k+1]
        print(f"Nearest neighbors of '{word}':")
        for n_idx in neighbors_idx:
            print(f"  {idx_to_word[n_idx]}: {similarities[n_idx]:.3f}")
        print()

def plot_embeddings(embeddings, word_to_idx, target_words, filename, title):
    """Plot embeddings with PCA and t-SNE."""
    indices = [word_to_idx[w] for w in target_words if w in word_to_idx]
    vectors = embeddings[indices]

    # PCA
    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(target_words):
        if word in word_to_idx:
            plt.scatter(reduced_pca[i, 0], reduced_pca[i, 1], color='blue')
            plt.text(reduced_pca[i, 0]+0.01, reduced_pca[i, 1]+0.01, word)
    plt.title(f"PCA: {title}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(filename.replace(".png", "_pca.png"))
    plt.close()

    # t-SNE (dynamic perplexity)
    perplexity = min(5, len(vectors) - 1)
    tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=perplexity)
    reduced_tsne = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(target_words):
        if word in word_to_idx:
            plt.scatter(reduced_tsne[i, 0], reduced_tsne[i, 1], color='red')
            plt.text(reduced_tsne[i, 0]+0.01, reduced_tsne[i, 1]+0.01, word)
    plt.title(f"t-SNE: {title}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.savefig(filename.replace(".png", "_tsne.png"))
    plt.close()

# -----------------------------
# Main analysis
# -----------------------------
def main():
    target_words = [
        "harry", "ron", "hermione", "hogwarts",
        "dumbledore", "snape", "magic", "wand"
    ]

    # --- With stop words ---
    print("=== With Stop Words ===")
    emb_with_stop = torch.load("embeddings_with_stopwords.pt").numpy()
    words_with_stop = load_text("data/HP1.txt", max_words=50000, remove_stopwords=False)
    word_to_idx_with_stop, idx_to_word_with_stop = build_vocab(words_with_stop)

    nearest_neighbors(emb_with_stop, word_to_idx_with_stop, idx_to_word_with_stop, target_words)
    plot_embeddings(emb_with_stop, word_to_idx_with_stop, target_words, 
                    "embeddings_with_stopwords.png", "Word Embeddings With Stop Words")
    print("PCA and t-SNE plots saved for embeddings with stop words.\n")

    # --- No stop words ---
    print("=== No Stop Words ===")
    emb_no_stop = torch.load("embeddings_no_stopwords.pt").numpy()
    words_no_stop = load_text("data/HP1.txt", max_words=50000, remove_stopwords=True)
    word_to_idx_no_stop, idx_to_word_no_stop = build_vocab(words_no_stop)

    nearest_neighbors(emb_no_stop, word_to_idx_no_stop, idx_to_word_no_stop, target_words)
    plot_embeddings(emb_no_stop, word_to_idx_no_stop, target_words, 
                    "embeddings_no_stopwords.png", "Word Embeddings Without Stop Words")
    print("PCA and t-SNE plots saved for embeddings without stop words.")

if __name__ == "__main__":
    main()
