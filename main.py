from RNN_Model import *
from p1 import *
import torch
import matplotlib.pyplot as plt
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

# Hyper-parameters for the model
input_size = len(dataset_train.vocab)
emb_size = 64
hidden_size = 64
output_size = len(dataset_train.classes)
eta = 0.001  # Increased learning rate
nb_epochs = 50  # More epochs

# Initialize model, loss function, and optimizer
model_manual = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)

loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_manual.parameters(), lr=eta)  # Use Adam

train_losses = []
val_accuracies = []

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
    
    val_acc = acc / len(dataset_test)
    avg_loss = total_loss / len(dataloader_train)
    
    # Store metrics
    train_losses.append(avg_loss)
    val_accuracies.append(val_acc)
    
    print(f'Epoch {n+1}/{nb_epochs}, Loss: {avg_loss:.4f}, Accuracy: {val_acc:.4f}')
    

# Save the model checkpoint
torch.save({
    'model_state_dict': model_manual.state_dict(),
    'vocab': dataset_train.vocab,
    'classes': dataset_train.classes
}, "rnn_model_checkpoint.pth")

print("Model saved successfully to rnn_model_checkpoint.pth")

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
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()