import torch
import torch.nn as nn

class CustomRNN_manual(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size,
                 pad_idx=None, use_residual=True, mask_idx=None):
        super(CustomRNN_manual, self).__init__()

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=pad_idx)
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.use_residual = use_residual
        self.hidden_size = hidden_size

        # id du token <mask> (obligatoire pour capter h_t_mask)
        self.mask_idx = mask_idx
        assert self.mask_idx is not None, "CustomRNN_manual: mask_idx doit être fourni."

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, inputs, mini_batch=False):
        """
        inputs: LongTensor [B, T]
        On renvoie la logit basée sur h_{t_mask} (première occurrence de <mask>).
        S'il n'y a pas de <mask> dans une séquence, fallback = dernier état.
        """
        B, T = inputs.size()
        device = inputs.device

        # positions du premier <mask> par séquence ; si absent -> dernier index (T-1)
        mask_bool = (inputs == self.mask_idx)                          # [B, T]
        has_mask = mask_bool.any(dim=1)                                # [B]
        first_pos = torch.where(
            has_mask,
            mask_bool.int().argmax(dim=1),                             # premier True
            torch.full((B,), T - 1, dtype=torch.long, device=device)   # fallback
        )                                                              # [B]

        # état initial
        hidden = torch.randn(B, self.hidden_size, device=device) * 0.01

        # conteneur pour h_{t_mask}
        masked_hidden = torch.zeros_like(hidden)
        captured = torch.zeros(B, dtype=torch.bool, device=device)

        # boucle temporelle
        for t in range(T):
            input_t = inputs[:, t]
            embedded = self.embedding(input_t)
            combined = torch.cat((embedded, hidden), dim=1)
            new_hidden = torch.tanh(self.layernorm(self.i2h(combined)))
            hidden = 0.5 * hidden + new_hidden if self.use_residual else new_hidden
            hidden = torch.clamp(hidden, -5, 5)

            # capturer l'état au bon pas (première occurrence)
            to_capture = (~captured) & (first_pos == t)
            if to_capture.any():
                masked_hidden[to_capture] = hidden[to_capture]
                captured[to_capture] = True

        # sécurité (devrait déjà être True partout)
        masked_hidden[~captured] = hidden[~captured]

        output = self.h2o(self.dropout(masked_hidden))
        return output, masked_hidden
