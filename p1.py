import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchtext.vocab import Vocab
from collections import Counter

import pandas as pd
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
            parts = line.rsplit(';', 1)
            if len(parts) == 2:
                text, emotion = parts
                texts.append(text)
                emotions.append(emotion)
    return texts, emotions

def undersample_dataset(texts, emotions):
    """
    Undersample the dataset to balance classes
    Returns: balanced_texts, balanced_emotions
    """
    # Create DataFrame-like structure
    data = list(zip(texts, emotions))
    
    # Count examples per class
    emotion_counts = Counter(emotions)
    print(f"Original class distribution: {emotion_counts}")
    
    # Find the minority class count
    min_count = min(emotion_counts.values())
    print(f"Undersampling: each class will have {min_count} examples")
    
    # Group by emotion and sample
    emotion_to_texts = {}
    for text, emotion in data:
        if emotion not in emotion_to_texts:
            emotion_to_texts[emotion] = []
        emotion_to_texts[emotion].append(text)
    
    # Undersample each class
    balanced_texts = []
    balanced_emotions = []
    
    for emotion, text_list in emotion_to_texts.items():
        # Sample min_count examples from each class
        sampled_texts = text_list[:min_count]  # Or use random.sample for randomness
        balanced_texts.extend(sampled_texts)
        balanced_emotions.extend([emotion] * min_count)
    
    print(f"Final dataset size: {len(balanced_texts)}")
    return balanced_texts, balanced_emotions

# Alternative version using random sampling
def undersample_dataset_random(texts, emotions, random_state=42):
    """
    Undersample with random sampling for each class
    """
    import random
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
    def __init__(self, texts, emotions, max_len=20, vocab=None, classes=None ,rare_threshold = 0):
        self.texts = texts
        self.emotions = emotions
        self.max_len = max_len
        self.rare_threshold = rare_threshold
        
        # Nếu vocab và classes được cung cấp, dùng chúng
        if vocab is not None and classes is not None:
            self.vocab = vocab
            self.classes = classes
            self.pad_idx = vocab.stoi["<pad>"]
            self.unk_idx = vocab.stoi["<unk>"]
        else:
            
            self.classes, self.vocab, self.pad_idx, self.unk_idx = self.computeClassesAndVocabs()

    def computeClassesAndVocabs(self):
        """
        Construit le vocabulaire et les classes, en retirant les mots rares (<rare_threshold occurrences)
        """

        # Build token list pour tout le dataset
        flattened_tokens = [token for tokens in yield_tokens(self.texts) for token in tokens]
        token_counts = Counter(flattened_tokens)
        
        if self.rare_threshold > 0 : 
            # Identifier les mots fréquents
            frequent_tokens = {token: count for token, count in token_counts.items() if count >= self.rare_threshold}
            print(f"Taille vocab avant filtrage : {len(token_counts)}, après filtrage des mots rares (<{self.rare_threshold} occurrences) : {len(frequent_tokens)}")
            
        # Spéciaux
        specials = ["<pad>", "<unk>"]
        
        # Construire le vocab avec seulement les tokens fréquents
        if self.rare_threshold > 0 : 
            vocab = Vocab(Counter(frequent_tokens), specials=specials)
        else : 
            vocab = Vocab(Counter(token_counts), specials=specials)
        pad_idx = vocab.stoi["<pad>"]
        unk_idx = vocab.stoi["<unk>"]

        # Vocab pour les classes / émotions
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

    
