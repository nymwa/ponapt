import torch.nn as nn

class SelfAttentionSubLayer(nn.Module):

    def __init__(
            self,
            d_model,
            nhead,
            dropout,
            attention_dropout):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout = attention_dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x,
            attn_mask,
            padding_mask):

        z = self.norm(x)
        z, _ = self.self_attn(
                z, z, z,
                attn_mask = attn_mask,
                key_padding_mask = padding_mask)
        x = x + self.dropout(z)
        return x

