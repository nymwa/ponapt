import torch.nn as nn

class FeedForwardSubLayer(nn.Module):

    def __init__(
            self,
            d_model,
            dim_feedforward,
            dropout):

        super().__init__()
        self.linear1 = nn.Linear(
                d_model,
                dim_feedforward)
        self.linear2 = nn.Linear(
                dim_feedforward,
                d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        z = self.norm(x)
        z = self.linear1(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.linear2(z)
        x = x + self.dropout(z)
        return x

