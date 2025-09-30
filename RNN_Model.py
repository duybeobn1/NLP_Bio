import torch.nn as nn
import torch.nn.functional as F

class CustomRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size=6):
        super(CustomRNN, self).__init__()

        self.i2e = nn.Linear(input_size, emb_size)
        self.hidden_size = hidden_size
        self.combined = nn.RNN(emb_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        #res = self.i2e(line_tensor)
        embedded = self.i2e(line_tensor)
        rnn_out, hidden = self.combined(embedded)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output



