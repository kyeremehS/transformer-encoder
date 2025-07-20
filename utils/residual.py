from enum import Enum

class NormType(Enum):
    LAYER_NORM = "layernorm"
    DYT = "dyt"

from torch import nn
from utils.DyT import DyT

class ResidualConnection(nn.Module):
    def __init__(self, size: int, dropout: float, norm_type: NormType):
        super().__init__()
        if norm_type == NormType.LAYER_NORM:
            self.norm = nn.LayerNorm(size)
        elif norm_type == NormType.DYT:
            self.norm = DyT(size)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

    
