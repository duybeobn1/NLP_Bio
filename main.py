
from RNN_Model import *
from p1 import *
import torch

# Data loading
train_texts, train_emotions = load_file("./dataset/train.txt")
test_texts, test_emotions = load_file("./dataset/test.txt")

#Hyperparamètres extra modèle
max_sequence_length = 20
batch_size = 10

# Dataset and DataLoader creation
dataset_train = EmotionDataset(train_texts, train_emotions, max_len=max_sequence_length)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Using vocab and classes from dataset_train for dataset_test
dataset_test = EmotionDataset(test_texts, test_emotions, max_len=max_sequence_length, 
                               vocab=dataset_train.vocab, classes=dataset_train.classes)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Hyper-Paramètres
input_size = len(dataset_train.vocab)
emb_size = 256
hidden_size = 128
output_size = len(dataset_train.classes)
eta = 0.001
nb_epochs = 20

#Définition Modèle
model_manual = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)

loss_func = torch.nn.CrossEntropyLoss()  # Change MSELoss to CrossEntropyLoss for classification
optim = torch.optim.SGD(model_manual.parameters(), lr=eta)

print(dataset_train)

# Training loop
for n in range(nb_epochs):
    model_manual.train()  # Set model to training mode
    for x, t in dataloader_train:
        optim.zero_grad()  # Clear gradients
        y, _ = model_manual(x, mini_batch = False)   # Forward pass
        loss = loss_func(y, t)  # Compute loss
        loss.backward()    # Backward pass
        optim.step()       # Update weights

    # Test loop
    model_manual.eval()  # Set model to evaluation mode
    acc = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for x, t in dataloader_test:
            y, _ = model_manual(x, mini_batch = False)
            acc += (torch.argmax(y, 1) == t).sum().item()
    print(f'Epoch {n+1}/{nb_epochs}, Accuracy: {acc / len(dataset_test)}')