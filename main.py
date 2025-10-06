from RNN_Model import *
from p1 import *
import torch

# Data loading
train_texts, train_emotions = load_file("./dataset/train.txt")
test_texts, test_emotions = load_file("./dataset/test.txt")

# Apply undersampling to training data
print("Before undersampling:")
print(f"Training set size: {len(train_texts)}")
train_texts_balanced, train_emotions_balanced = undersample_dataset_random(
    train_texts, train_emotions
)

# Hyperparamètres
max_sequence_length = 20
batch_size = 10

# Dataset and DataLoader creation with balanced data
dataset_train = EmotionDataset(train_texts_balanced, train_emotions_balanced, max_len=max_sequence_length)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Test dataset uses original test distribution
dataset_test = EmotionDataset(test_texts, test_emotions, max_len=max_sequence_length, 
                               vocab=dataset_train.vocab, classes=dataset_train.classes)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Hyper-Paramètres
input_size = len(dataset_train.vocab)
emb_size = 128  # Increased
hidden_size = 128
output_size = len(dataset_train.classes)
eta = 0.001  # Increased learning rate
nb_epochs = 50  # More epochs

# Définition Modèle
model_manual = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)

loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_manual.parameters(), lr=eta)  # Use Adam

# Training loop with improvements
for n in range(nb_epochs):
    model_manual.train()
    total_loss = 0
    for x, t in dataloader_train:
        optim.zero_grad()
        y, _ = model_manual(x, mini_batch=False)
        loss = loss_func(y, t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_manual.parameters(), max_norm=1.0)
        optim.step()
        total_loss += loss.item()
    
    # Validation
    model_manual.eval()
    acc = 0
    with torch.no_grad():
        for x, t in dataloader_test:
            y, _ = model_manual(x, mini_batch=False)
            acc += (torch.argmax(y, 1) == t).sum().item()
    
    # Print training and validation loss
    val_acc = acc / len(dataset_test)
    print(f'Epoch {n+1}/{nb_epochs}, Loss: {total_loss/len(dataloader_train):.4f}, Accuracy: {val_acc:.4f}')
    