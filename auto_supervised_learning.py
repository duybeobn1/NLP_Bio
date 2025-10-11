from RNN_Model_auto_supervised import *          # load_file, undersample_dataset_random, EmotionDataset, tokenizer, etc.
from p1 import *                 # CustomRNN_manual
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from collections import Counter
import random
import matplotlib.pyplot as plt

# -----------------------------
# Dataset pour prédiction de mot masqué
# -----------------------------
class MaskedWordDataset(Dataset):
    """
    Pour chaque phrase:
      - choisit un token éligible aléatoire
      - le remplace par <mask>
      - cible = id du mot original
    """
    def __init__(self, texts, vocab, pad_idx, unk_idx, mask_idx, max_len=20, seed=42):
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.mask_idx = mask_idx
        self.max_len = max_len
        random.seed(seed)

        self.samples = []  # (masked_ids, target_id)
        for text in texts:
            toks = tokenizer(text)
            if not toks:
                continue
            # positions éligibles (on évite les spéciaux si déjà présents dans les données)
            eligible = [i for i, tok in enumerate(toks) if tok not in ("<pad>", "<unk>", "<mask>")]
            if not eligible:
                continue
            pos = random.choice(eligible)
            true_tok = toks[pos]
            target_id = vocab.stoi.get(true_tok, unk_idx)

            ids = [vocab.stoi.get(tok, unk_idx) for tok in toks]
            ids[pos] = mask_idx

            # pad/tronque
            if len(ids) < max_len:
                ids = ids + [pad_idx] * (max_len - len(ids))
            else:
                ids = ids[:max_len]

            self.samples.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.long)
            ))

        if len(self.samples) == 0:
            raise ValueError("Aucun échantillon valide après masquage. Vérifie ton dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_masked(batch):
    x = torch.stack([b[0] for b in batch], dim=0)  # [B, T]
    t = torch.stack([b[1] for b in batch], dim=0)  # [B]
    return x, t

# -----------------------------
# Utilitaire: vocab avec <mask>
# -----------------------------
def build_vocab_with_mask(texts, base_specials=("s<pad>", "s<unk>", "s<mask>")):
    """
    Construit un Vocab TorchText avec <pad>, <unk>, <mask>.
    NB: on reconstruit un vocab (indépendant de EmotionDataset)
    """
    tokens = [tok for sent in texts for tok in tokenizer(sent)]
    counter = Counter(tokens)
    specials = ["<pad>", "<unk>", "<mask>"]
    vocab = Vocab(counter, specials=specials)
    pad_idx = vocab.stoi["<pad>"]
    unk_idx = vocab.stoi["<unk>"]
    mask_idx = vocab.stoi["<mask>"]
    return vocab, pad_idx, unk_idx, mask_idx

# -----------------------------
# Data loading (comme ton format)
# -----------------------------
train_texts, train_emotions = load_file("./dataset/train.txt")
test_texts, test_emotions = load_file("./dataset/test.txt")

# Apply undersampling to training data (on garde la même étape pour respecter ton format)
print("Before undersampling:")
print(f"Training set size: {len(train_texts)}")
train_texts_balanced, train_emotions_balanced = undersample_dataset_random(
    train_texts, train_emotions
)

# Hyperparamètres
max_sequence_length = 20
batch_size = 10

# On crée quand même EmotionDataset pour rester aligné à ton squelette,
# mais on va utiliser un vocab dédié avec <mask> pour la tâche de MLM.
dataset_train_tmp = EmotionDataset(train_texts_balanced, train_emotions_balanced, max_len=max_sequence_length)

# Vocab dédié avec <mask> (reconstruit à partir des textes d'entraînement équilibrés)
vocab, pad_idx, unk_idx, mask_idx = build_vocab_with_mask(train_texts_balanced)
vocab_size = len(vocab)

# Dataset/Dataloader masqués (train/test)
dataset_train = MaskedWordDataset(
    texts=train_texts_balanced,
    vocab=vocab,
    pad_idx=pad_idx,
    unk_idx=unk_idx,
    mask_idx=mask_idx,
    max_len=max_sequence_length,
)

dataset_test = MaskedWordDataset(
    texts=test_texts,
    vocab=vocab,
    pad_idx=pad_idx,
    unk_idx=unk_idx,
    mask_idx=mask_idx,
    max_len=max_sequence_length,
)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_masked)
dataloader_test  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, collate_fn=collate_masked)

# Hyper-parameters for the model
input_size = vocab_size              # != len(dataset_train_tmp.vocab), on prend le vocab avec <mask>
emb_size = 64
hidden_size = 64
output_size = vocab_size             # sortie = vocab complet pour prédire le mot
eta = 0.001
nb_epochs = 50

# Initialize model, loss function, and optimizer
model_manual = CustomRNN_manual(
    input_size=vocab_size,
    emb_size=128,
    hidden_size=128,
    output_size=vocab_size,
    pad_idx=pad_idx,
    use_residual=True,
    mask_idx=mask_idx,   # <-- important
)

loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_manual.parameters(), lr=eta)

train_losses = []
val_accuracies = []

# Boucle d'entraînement et validation (accuracy = top-1 sur le mot masqué)
for n in range(nb_epochs):
    model_manual.train()
    total_loss = 0.0
    total_items = 0

    for x, t in dataloader_train:
        optim.zero_grad()
        y, _ = model_manual(x, mini_batch=False)  # y: [B, |V|]
        loss = loss_func(y, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_manual.parameters(), max_norm=1.0)
        optim.step()
        total_loss += loss.item() * x.size(0)
        total_items += x.size(0)

    # Validation (top-1 accuracy)
    model_manual.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, t in dataloader_test:
            y, _ = model_manual(x, mini_batch=False)  # [B, |V|]
            pred = torch.argmax(y, dim=1)             # [B]
            correct += (pred == t).sum().item()
            total += t.size(0)

    val_acc = correct / max(1, total)
    avg_loss = total_loss / max(1, total_items)

    train_losses.append(avg_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch {n+1}/{nb_epochs}, Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}')

# Save the model checkpoint (on sauvegarde aussi le vocab)
# Save the model checkpoint

torch.save({
    "model_state_dict": model_manual.state_dict(),
    "vocab": vocab,          # torchtext.vocab.Vocab (contient .itos / .stoi / __len__)
    "classes": vocab,        # même objet ; len(checkpoint['classes']) == len(vocab)
    # Optionnel si tu veux :
    "pad_idx": pad_idx,
    "unk_idx": unk_idx,
    "mask_idx": mask_idx,
}, "rnn_model_checkpoint_auto_supervised.pth")

print("Model saved successfully to rnn_model_checkpoint_auto_supervised.pth")
# Visualization of training loss and validation accuracy
plt.figure(figsize=(10,4))

# Plot Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label='Training Loss')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1,2,2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title("Validation Accuracy (Top-1) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
