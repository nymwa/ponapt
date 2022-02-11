import torch.nn as nn
from .transformer.embedding import TransformerEmbedding
from .transformer.lm import TransformerLM

class LM(nn.Module):

    def __init__(
            self,
            d_vocab,
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            word_dropout,
            attention_dropout,
            activation_dropout,
            num_layers,
            padding_idx = 0,
            max_len = 64):

        super().__init__()

        self.embedding = TransformerEmbedding(
                d_vocab,
                d_model,
                dropout,
                word_dropout,
                padding_idx,
                max_len)

        self.lm = TransformerLM(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                attention_dropout,
                activation_dropout,
                num_layers)

        self.fc = nn.Linear(d_model, d_vocab, bias = False)

    def forward(self, batch):

        x = self.embedding(
                batch.inputs,
                position_ids = batch.position)

        x = self.lm(
                x,
                attn_mask = batch.mask,
                padding_mask = batch.padding)

        x = self.fc(x)
        return x

