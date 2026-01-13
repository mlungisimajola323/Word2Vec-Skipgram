import re

STOP_WORDS = {
    "the", "and", "to", "of", "a", "in", "that", "it", "is",
    "was", "he", "she", "they", "his", "her", "you", "for",
    "on", "with", "as", "at", "by", "an", "be", "this"
}

def load_text(path, max_words=None, remove_stopwords=False):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()

    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()

    if remove_stopwords:
        words = [w for w in words if w not in STOP_WORDS]

    if max_words:
        words = words[:max_words]

    return words

def build_vocab(words):
    """
    Builds two dictionaries:
    - word_to_idx: maps each unique word to a unique integer index
    - idx_to_word: maps each index back to the word
    """
    unique_words = list(dict.fromkeys(words))  # preserves order of first appearance
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word
