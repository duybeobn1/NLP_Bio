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

# Data loading
train_texts, train_emotions = load_file("./dataset/train.txt")

# Build token list for Vocab
flattened_tokens = [token for tokens in yield_tokens(train_texts) for token in tokens]
counter = Counter(flattened_tokens)
specials = ["<pad>", "<unk>"]
vocab = Vocab(counter, specials=specials)
pad_idx = vocab.stoi["<pad>"]
unk_idx = vocab.stoi["<unk>"]

# Vocab for emotions/classes
class_names = sorted(set(train_emotions))
classes = {e: i for i, e in enumerate(class_names)}

class EmotionDataset(Dataset):
    def __init__(self, texts, emotions, vocab, classes, max_len=20):
        self.texts = texts
        self.emotions = emotions
        self.vocab = vocab
        self.classes = classes
        self.max_len = max_len
        self.pad_idx = vocab.stoi["<pad>"]
        self.unk_idx = vocab.stoi["<unk>"]

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

# Hyperparameters
max_sequence_length = 20
batch_size = 10

# Dataset and DataLoader creation
dataset = EmotionDataset(train_texts, train_emotions, vocab, classes, max_len=max_sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for inputs, targets in dataloader:
    print("Inputs (batch_size, seq_len, vocab_size):", inputs.shape)
    print("Targets (batch_size):", targets.shape)
    print("Targets (sample values):", targets[:3])
    break


# For mapping label indices back to emotion, if needed:
idx2emotion = {i: emotion for emotion, i in classes.items()}
