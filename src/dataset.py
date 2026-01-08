import torch
from torch.utils.data import Dataset

class SkipGramDataset(Dataset):
    def __init__(self, words, word_to_idx, window_size=2):
        self.data = []
        self.word_to_idx = word_to_idx

        for i, center_word in enumerate(words):
            center_idx = word_to_idx[center_word]

            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_word = words[j]
                    context_idx = word_to_idx[context_word]
                    self.data.append((center_idx, context_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.tensor(center), torch.tensor(context)
