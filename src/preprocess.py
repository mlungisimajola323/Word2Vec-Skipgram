import string

def load_text(file_path, max_words=None):
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    # Split by whitespace
    words = text.split()

    if max_words is not None:
        words = words[:max_words]

    return words

def build_vocab(words):
    
    word_to_idx = {}
    idx_to_word = {}

    for word in words:
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    return word_to_idx, idx_to_word