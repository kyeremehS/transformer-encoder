from utils.DyT import DyT
from utils.residual import ResidualConnection
import torch.nn as nn

from models.attention import MultiheadAttention
from models.feedforward import PositionWiseFeed
from utils.residual import NormType


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm_type=NormType.LAYER_NORM):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionWiseFeed(d_model, d_ff, dropout)

        self.residual_attn = ResidualConnection(d_model, dropout, norm_type)
        self.residual_ff = ResidualConnection(d_model, dropout, norm_type)

    def forward(self, x, mask=None):
        # Apply multi-head attention + residual connection + normalization
        x = self.residual_attn(x, lambda x_: self.self_attn(x_, mask))

        # Apply feedforward network + residual connection + normalization
        x = self.residual_ff(x, self.feedforward)



