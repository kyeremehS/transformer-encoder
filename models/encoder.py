from models.encoder_layer import TransformerEncoderLayer
import torch.nn as nn
from utils.DyT import DyT
from utils.residual import NormType


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,d_ff, dropout,norm_type):
        super.__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_type == NormType.LAYER_NORM else DyT(d_model)


    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x