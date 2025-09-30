import torch
import torch.nn as nn
import torch.nn.functional as F
from p1 import *


class CustomRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size=6):
        super(CustomRNN, self).__init__()
        self.i2e = nn.Linear(input_size, emb_size) # input as input_size with the same size of vocab (one-hot) -> reduce to emb_size( numbers of dimensions )
        self.hidden_size = hidden_size 
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor shape: (batch_size, seq_len, input_size)
        embedded = self.i2e(line_tensor)            # (batch, seq_len, emb_size) 
        rnn_out, hidden = self.rnn(embedded)        # rnn_out: (batch, seq_len, hidden_size)
        output = self.h2o(hidden.squeeze(0))        # hidden.shape: (num_layers=1, batch, hidden_size)
                                                    # take the last hidden state
        output = self.softmax(output)                # (batch, output_size)
        return output

# Instanciation modèle (exemple)
input_size = len(vocab)   # taille vocabulaire
emb_size = 256
hidden_size = 128
output_size = len(classes)

model = CustomRNN(input_size, emb_size, hidden_size, output_size)


### 1 mot
model.eval()
with torch.no_grad():
    one_hot_sentence, label = dataset[0]  # (seq_len, vocab_size)
    input_word = one_hot_sentence[0].unsqueeze(0).unsqueeze(0)  # Add dim seq_len: shape (1, 1, vocab_size)

    # hidden (num_layers=1, batch_size=1, hidden_size)
    hidden = torch.zeros(1, 1, model.hidden_size)  
    
    # Input for model
    output, hidden_out = model.rnn(model.i2e(input_word), hidden)
    
    # Last hidden (batch_size=1)
    logits = model.h2o(hidden_out.squeeze(0))
    probs = model.softmax(logits)
    
    print("Output log-probs 1-word:", probs)

    # probs: output của logsoftmax với shape (batch_size=1, num_classes=6)
    predicted_class_idx = probs.argmax(dim=1).item()

    print("Guessing (class index):", predicted_class_idx)

    # For convenience, map back to emotion names
    idx2emotion = {i: e for e, i in classes.items()}
    print("Guessing emotion:", idx2emotion.get(predicted_class_idx, "Unknown"))

