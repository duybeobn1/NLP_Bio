import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchtext.vocab import Vocab
from collections import Counter

def tokenizer(text):
    """Simple tokenizer - modify for advanced needs"""
    return text.lower().split()

def yield_tokens(texts):
    """Generator for tokens in all texts"""
    for text in texts:
        yield tokenizer(text)

def load_file(file):
    """Load (text, emotion) pairs from file with the format: <sentence> <emotion>"""
    texts = []
    emotions = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                text, emotion = parts
                texts.append(text)
                emotions.append(emotion)
    return texts, emotions

class EmotionDataset(Dataset):
    def __init__(self, texts, emotions, max_len=20, vocab=None, classes=None):
        self.texts = texts
        self.emotions = emotions
        self.max_len = max_len
        
        # Nếu vocab và classes được cung cấp, dùng chúng
        if vocab is not None and classes is not None:
            self.vocab = vocab
            self.classes = classes
            self.pad_idx = vocab.stoi["<pad>"]
            self.unk_idx = vocab.stoi["<unk>"]
        else:
            # Nếu không, tính toán mới
            self.classes, self.vocab, self.pad_idx, self.unk_idx = self.computeClassesAndVocabs()

    def computeClassesAndVocabs(self):
        # Build token list for Vocab
        flattened_tokens = [token for tokens in yield_tokens(self.texts) for token in tokens]
        counter = Counter(flattened_tokens)
        specials = ["<pad>", "<unk>"]
        vocab = Vocab(counter, specials=specials)
        pad_idx = vocab.stoi["<pad>"]
        unk_idx = vocab.stoi["<unk>"]

        # Vocab for emotions/classes
        class_names = sorted(set(self.emotions))
        classes = {e: i for i, e in enumerate(class_names)}

        return classes, vocab, pad_idx, unk_idx


    def __len__(self):
        return len(self.texts)

    def encode_and_pad(self, text):
        tokens = tokenizer(text)
        indices = [self.vocab.stoi.get(token, self.unk_idx) for token in tokens]
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [self.pad_idx] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        indices = self.encode_and_pad(self.texts[idx])
        one_hot_tensor = one_hot(indices, num_classes=len(self.vocab)).float()
        emotion_label = self.classes.get(self.emotions[idx], -1)
        return one_hot_tensor, emotion_label

