import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchtext.vocab import Vocab
from collections import Counter
import random
import pandas as pd

def tokenizer(text):
    """lowercase and split by space"""
    return text.lower().split()

def yield_tokens(texts):
    """generate tokens from a list of texts"""
    for text in texts:
        yield tokenizer(text)

def load_file(file):
    """Load (text, emotion) pairs from file with the format: <sentence> ; <emotion>"""
    texts = []
    emotions = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(';', 1)
            if len(parts) == 2:
                text, emotion = parts
                texts.append(text)
                emotions.append(emotion)
    return texts, emotions

# undersample dataset to balance classes using random sampling
def undersample_dataset_random(texts, emotions, random_state=42):
    """
    undersample with random sampling for each class to balance the dataset for training
    """
    random.seed(random_state)
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({'text': texts, 'emotion': emotions})
    
    # Find the minority class count
    min_count = df['emotion'].value_counts().min()
    print(f"Undersampling: each class will have {min_count} examples")
    
    # Sample from each class
    df_balanced = df.groupby('emotion').apply(
        lambda x: x.sample(n=min_count, random_state=random_state)
    ).reset_index(drop=True)
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=random_state)
    
    print(f"Original size: {len(df)}, Balanced size: {len(df_balanced)}")
    print(f"Balanced distribution:\n{df_balanced['emotion'].value_counts()}")
    
    return df_balanced['text'].tolist(), df_balanced['emotion'].tolist()

class EmotionDataset(Dataset):
    """
    Dataset for emotion classification. Each item is a tuple (text_indices, emotion_label).
    texts: list of sentences
    emotions: list of corresponding emotion labels
    max_len: maximum length for padding/truncating sequences
    vocab: precomputed Vocab object (optional)
    classes: precomputed class mapping (optional)
    """
    def __init__(self, texts, emotions, max_len=20, vocab=None, classes=None):
        self.texts = texts
        self.emotions = emotions
        self.max_len = max_len
        
        # if vocab and classes are provided, use them; otherwise compute from data
        if vocab is not None and classes is not None:
            self.vocab = vocab
            self.classes = classes
            self.pad_idx = vocab.stoi["<pad>"]
            self.unk_idx = vocab.stoi["<unk>"]
        else:
            
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
        emotion_label = self.classes.get(self.emotions[idx], -1)
        return indices, emotion_label  # Return indices instead of one-hot tensors

    
