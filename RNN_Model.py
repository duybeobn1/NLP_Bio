import torch
import torch.nn as nn

class CustomRNN_manual(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size):
        super(CustomRNN_manual, self).__init__()
        # Linear layer to project one-hot (vocab) -> embedding
        self.i2e = nn.Linear(input_size, emb_size)
        # Recurrence: input is concatenation (emb + hidden)
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(emb_size + hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, inputs, mini_batch=True):
        """
        Args:
            inputs: 
                - mini_batch=True: (batch_size, input_size) for single step
                - mini_batch=False: (batch_size, seq_len, input_size) for sequences
            mini_batch: bool, whether to process single step or full sequence
        
        Returns:
            output: (batch_size, output_size) - raw logits
            hidden: (batch_size, hidden_size) - final hidden state
        """
        device = inputs.device
        
        if mini_batch:
            # Single time-step processing
            batch_size = inputs.size(0)
            hidden = torch.zeros(batch_size, self.hidden_size, device=device)
            
            embedded = self.i2e(inputs)                           # (batch_size, emb_size)
            combined = torch.cat((embedded, hidden), dim=1)       # (batch_size, emb+hid)
            hidden = torch.tanh(self.i2h(combined))               # (batch_size, hidden_size)
            output = self.i2o(combined)                           # (batch_size, output_size)
            
            return output, hidden

        else:
            # Full sequence processing
            batch_size = inputs.size(0)
            seq_len = inputs.size(1)
            hidden = torch.zeros(batch_size, self.hidden_size, device=device)
            
            for t in range(seq_len):
                input_t = inputs[:, t, :]                         # (batch_size, input_size)
                embedded = self.i2e(input_t)                      # (batch_size, emb_size)
                combined = torch.cat((embedded, hidden), dim=1)   # (batch_size, emb+hid)
                hidden = torch.tanh(self.i2h(combined))           # (batch_size, hidden_size)
                output = self.i2o(combined)                       # (batch_size, output_size)
            
            # Return logits for last time-step and final hidden state
            return output, hidden