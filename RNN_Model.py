import torch
import torch.nn as nn
import torch.nn.functional as F
from p1 import *


class CustomRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size=6):
        super(CustomRNN, self).__init__()
        self.i2e = nn.Linear(input_size, emb_size)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        # line_tensor shape: (batch_size, seq_len, input_size)
        embedded = self.i2e(line_tensor)            # (batch, seq_len, emb_size)
        rnn_out, hidden = self.rnn(embedded)        # rnn_out: (batch, seq_len, hidden_size)
        output = self.h2o(hidden.squeeze(0))        # hidden.shape: (num_layers=1, batch, hidden_size)
                                                    # lấy hidden state cuối cùng của layer ẩn (batch, hidden_size)
        output = self.softmax(output)                # (batch, output_size)
        return output

# Instanciation modèle (exemple)
input_size = len(vocab)   # taille vocabulaire
emb_size = 256
hidden_size = 128
output_size = len(classes)

model = CustomRNN(input_size, emb_size, hidden_size, output_size)


batch_input = torch.randn(10, 20, input_size)

output = model(batch_input)
print("Output shape:", output.shape)   # Kỳ vọng: (10, 6)