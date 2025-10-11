import torch
from torch.utils.data import DataLoader
from RNN_Model import CustomRNN_manual
from p1 import EmotionDataset, load_file  # assuming p1.py contains dataset utils
import torchtext.vocab



torch.serialization.add_safe_globals([torchtext.vocab.Vocab])
checkpoint = torch.load("rnn_model_checkpoint.pth")

# === Recreate model with same structure ===
input_size = len(checkpoint['vocab'])
emb_size = 128
hidden_size = 128
output_size = len(checkpoint['classes'])

model_loaded = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)
model_loaded.load_state_dict(checkpoint['model_state_dict'])
model_loaded.eval()

print(" Model loaded successfully!")


# === Load val data ===
val_texts, val_emotions = load_file("./dataset/val.txt")

# === Create dataset using saved vocab and classes ===
dataset_val = EmotionDataset(
    val_texts,
    val_emotions,
    max_len=20,
    vocab=checkpoint['vocab'],
    classes=checkpoint['classes']
)

dataloader_val = DataLoader(dataset_val, batch_size=10, shuffle=False)


# === Evaluate ===
correct = 0
total = 0

with torch.no_grad():
    for x, t in dataloader_val:
        y, _ = model_loaded(x)
        preds = torch.argmax(y, dim=1)
        correct += (preds == t).sum().item()
        total += t.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy*100:.2f}%")
