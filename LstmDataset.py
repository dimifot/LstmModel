#from tokenizerFile import tokenizer
import torch
from torch.utils.data import Dataset
from tokenizerFile import tokenizer, py_tokenizer
class CodeDataset(Dataset):
    def __init__(self, snippets, labels, seq_length):
        self.tokenized_snippets = [tokenizer(snippet) for snippet in snippets]
        print(f"tokenized_snippets size: {len(self.tokenized_snippets)}")
        self.tokens = sorted(list(set(token for snippet in self.tokenized_snippets for token in snippet)))
        print(f"tokens size: {self.tokens}")
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.tokens)
        print(self.vocab_size)
        self.snippets = snippets
        self.labels = labels
        self.seq_length = seq_length
        self.data = self.prepare_data(self.tokenized_snippets, labels, seq_length)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Number of sequences: {len(self.data)}")
    def prepare_data(self, tokenized_snippets, labels, seq_length):
        sequences = []
        for snippet, label in zip(tokenized_snippets, labels):
            for i in range(0, len(snippet) - seq_length):
                seq = snippet[i:i + seq_length]
                sequences.append((seq, label))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, label = self.data[index]
        seq_idx = torch.tensor([self.token2idx[token] for token in seq], dtype=torch.long)
        label_idx = torch.tensor(label, dtype=torch.long)
        return seq_idx, label_idx

class CodeDataset_py(Dataset):
    def __init__(self, snippets, labels, seq_length):
        self.tokenized_snippets = [py_tokenizer(snippet) for snippet in snippets]
        print(f"tokenized_snippets size: {len(self.tokenized_snippets)}")
        self.tokens = sorted(list(set(token for snippet in self.tokenized_snippets for token in snippet)))
        print(f"tokens size: {self.tokens}")
        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.tokens)
        print(self.vocab_size)
        self.snippets = snippets
        self.labels = labels
        self.seq_length = seq_length
        self.data = self.prepare_data(self.tokenized_snippets, labels, seq_length)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Number of sequences: {len(self.data)}")
    def prepare_data(self, tokenized_snippets, labels, seq_length):
        sequences = []
        for snippet, label in zip(tokenized_snippets, labels):
            for i in range(0, len(snippet) - seq_length):
                seq = snippet[i:i + seq_length]
                sequences.append((seq, label))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, label = self.data[index]
        seq_idx = torch.tensor([self.token2idx[token] for token in seq], dtype=torch.long)
        label_idx = torch.tensor(label, dtype=torch.long)
        return seq_idx, label_idx
