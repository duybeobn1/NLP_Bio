
from RNN_Model import *
from p1 import *
import torch

# Data loading
train_texts, train_emotions = load_file("./dataset/train.txt")
test_texts, test_emotions = load_file("./dataset/test.txt")

# Dataset and DataLoader creation
dataset_train = EmotionDataset(train_texts, train_emotions, max_len=max_sequence_length)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = EmotionDataset(test_texts, test_emotions, max_len=max_sequence_length)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Hyper-Paramètres
input_size = len(dataset_train.vocab)
emb_size = 256
hidden_size = 128
output_size = len(dataset_train.classes)

#Définition Modèle
model_manual = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)


print(dataset_train)


"""
# Training loop
for n in range(nb_epochs):
    model.train()  # Set model to training mode
    for x, t in train_loader:
        optim.zero_grad()  # Clear gradients
        y = model(x)       # Forward pass
        loss = loss_func(y, t)  # Compute loss
        loss.backward()    # Backward pass
        optim.step()       # Update weights

    # Test loop
    model.eval()  # Set model to evaluation mode
    acc = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for x, t in test_loader:
            y = model(x)
            acc += (torch.argmax(y, 1) == t).item()
    print(f'Epoch {n+1}/{nb_epochs}, Accuracy: {acc / len(data_test)}')
"""