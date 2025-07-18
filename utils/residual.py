import torch.nn as nn
from utils.DyT import  DyT
class ResidualConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
    


class ResidualConnectionDyT(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = DyT(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

    
