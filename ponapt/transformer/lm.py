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

        self.num_layers = num_layers
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
            padding_mask = None,
            indices = None):

        if indices is None:
            indices = range(self.num_layers)

        for index in indices:
            layer = self.layers[index]
            x = layer(
                x,
                attn_mask = attn_mask,
                padding_mask = padding_mask)

        return x

