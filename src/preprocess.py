import string

def load_text(file_path, max_words=None):
    """
    Reads text from a file and returns a cleaned list of words.
    
    Args:
        file_path (str): Path to the text file
        max_words (int, optional): Limit number of words for faster experiments
    
    Returns:
        List[str]: List of cleaned words
    """
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
