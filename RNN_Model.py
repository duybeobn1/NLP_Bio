import torch
import torch.nn as nn
import torch.nn.functional as F
from p1 import *
class CustomRNN_manual(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size=6):
        super(CustomRNN_manual, self).__init__()
        self.i2e = nn.Linear(input_size, emb_size)  # Embedding
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)  # New hidden
        self.i2o = nn.Linear(emb_size + hidden_size, output_size)  # output
        self.hidden_size = hidden_size
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, input_size)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        hidden = torch.zeros(batch_size, self.hidden_size)

        for t in range(seq_len):
            input_t = inputs[:, t, :]  # (batch_size, input_size)
            embedded = self.i2e(input_t)   # (batch_size, emb_size)
            combined = torch.cat((embedded, hidden), dim=1)  # (batch_size, emb_size + hidden_size)

            hidden = torch.tanh(self.i2h(combined))  # hidden , new

            output = self.i2o(combined)               # output raw
            output = self.softmax(output)             # probabilites log

        return output  # last output

# Thí dụ khởi tạo model
input_size = len(vocab)
emb_size = 256
hidden_size = 128
output_size = len(classes)

model_manual = CustomRNN_manual(input_size, emb_size, hidden_size, output_size)

# 1-word
# model_manual.eval()
# with torch.no_grad():
#     one_hot_sentence, label = dataset[0]
#     input_word = one_hot_sentence[0].unsqueeze(0).unsqueeze(0) # shape (1, 1, vocab_size)

#     output = model_manual(input_word)

#     predicted_class = output.argmax(dim=1).item()

#     idx2emotion = {i:e for e,i in classes.items()}
#     print("Prediction:", idx2emotion.get(predicted_class, "Unknown"))



model_manual.eval()
with torch.no_grad():
    one_hot_sentence, label = dataset[0]  # (seq_len, vocab_size)
    hidden = torch.zeros(1, model_manual.hidden_size)
    for t in range(one_hot_sentence.shape[0]):
        input_word = one_hot_sentence[t].unsqueeze(0)  # (1, vocab_size)
        embedded = model_manual.i2e(input_word)
        combined = torch.cat((embedded, hidden), dim=1)
        hidden = torch.tanh(model_manual.i2h(combined))
        
        output = model_manual.i2o(combined)
        output = model_manual.softmax(output)
    
    pred_class = output.argmax(dim=1).item()
    print("Prediction récursive 1 phrase:", pred_class)
    idx2emotion = {i:e for e,i in classes.items()}
    print("Emotion:", idx2emotion.get(pred_class, "Unknown"))



