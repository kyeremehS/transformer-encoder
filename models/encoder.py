from models.encoder_layer import TransformerEncoderLayer
import torch.nn as nn
from utils.DyT import DyT
from utils.residual import NormType
from utils.postional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout, norm_type):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, norm_type)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_type == NormType.LAYER_NORM else DyT(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["d_model"])
        self.pe = PositionalEncoding(config["d_model"], config["max_len"])
        norm_type = NormType.LAYER_NORM if not config["use_dyt"] else NormType.DYT
        self.encoder = TransformerEncoder(config["num_layers"], config["d_model"], config["num_heads"],
                                          config["d_ff"], config["dropout"], norm_type)
        self.classifier = nn.Linear(config["d_model"], config["num_classes"])
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pe(x)         # Add positional encoding
        x = self.encoder(x)    # Apply transformer layers
        return self.classifier(x[:, 0])  # Use first token for classification

