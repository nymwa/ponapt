import torch.nn as nn
from .lm_layer import TransformerLMLayer

class TransformerLM(nn.Module):

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            attention_dropout,
            activation_dropout,
            num_layers):

        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLMLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                attention_dropout,
                activation_dropout)
            for _
            in range(num_layers)])

    def forward(
            self,
            x,
            attn_mask = None,
            padding_mask = None):

        for layer in self.layers:
            x = layer(
                    x,
                    attn_mask = attn_mask,
                    padding_mask = padding_mask)

        return x

