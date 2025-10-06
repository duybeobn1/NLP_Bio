import torch
import torch.nn as nn

class CustomRNN_manual(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, pad_idx=None, use_residual=True):
        super(CustomRNN_manual, self).__init__()
        
        # Embedding layer (with padding_idx if available)
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=pad_idx)
        
        # Core RNN transformation
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        # Regularization and normalization
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.use_residual = use_residual
        
        self.hidden_size = hidden_size
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Xavier initialization for weights and zero for biases
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, inputs, mini_batch=False):
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Small random initialization for hidden state
        hidden = torch.randn(batch_size, self.hidden_size, device=device) * 0.01
        
        # Process the sequence token by token
        for t in range(inputs.size(1)):
            input_t = inputs[:, t]
            embedded = self.embedding(input_t)
            
            # Concatenate input and hidden state
            combined = torch.cat((embedded, hidden), dim=1)
            
            # Compute new hidden state with normalization
            new_hidden = torch.tanh(self.layernorm(self.i2h(combined)))
            
            # Optional residual connection to improve gradient flow
            if self.use_residual:
                hidden = 0.5 * hidden + new_hidden
            else:
                hidden = new_hidden
            
            # Clamp to prevent explosion
            hidden = torch.clamp(hidden, -5, 5)
        
        # Apply dropout on the final hidden state
        output = self.h2o(self.dropout(hidden))
        return output, hidden
