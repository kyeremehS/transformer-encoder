# Transformer Encoder

A PyTorch implementation of a Transformer encoder with support for both standard Layer Normalization and a novel Dynamic Tanh (DyT) normalization technique.

## Features

- **Multi-Head Attention**: Parallel attention heads for capturing different types of relationships
- **Position-wise Feed Forward Networks**: Two-layer fully connected networks with ReLU activation
- **Flexible Normalization**: Support for both Layer Normalization and Dynamic Tanh (DyT) normalization
- **Positional Encoding**: Sinusoidal positional embeddings for sequence understanding
- **Residual Connections**: Skip connections with dropout for stable training

## Architecture

The transformer encoder consists of the following key components:

### Core Models
- [`MultiheadAttention`](models/attention.py) - Multi-head self-attention mechanism
- [`SelfAttention`](models/attention.py) - Single-head self-attention implementation
- [`PositionWiseFeed`](models/feedforward.py) - Position-wise feed forward network
- [`TransformerEncoderLayer`](models/encoder.py) - Complete encoder layer combining attention and feed forward

### Utilities
- [`DyT`](utils/DyT.py) - Dynamic Tanh normalization layer (novel alternative to Layer Norm)
- [`ResidualConnection`](utils/residual.py) - Residual connections with configurable normalization
- [`PositionalEncoding`](utils/postional_encoding.py) - Sinusoidal positional embeddings

## Project Structure

```
transformer-encoder/
├── models/
│   ├── attention.py      # Self-attention and multi-head attention
│   ├── encoder.py        # Transformer encoder layer
│   └── feedforward.py    # Position-wise feed forward network
├── utils/
│   ├── DyT.py           # Dynamic Tanh normalization
│   ├── residual.py      # Residual connections
│   └── postional_encoding.py  # Positional encoding
└── README.md
```

## Key Components

### Dynamic Tanh (DyT) Normalization

A novel normalization technique implemented in [`DyT`](utils/DyT.py) that uses:
- Learnable alpha parameter for controlling tanh saturation
- Element-wise weight and bias parameters
- Tanh activation for bounded outputs

### Multi-Head Attention

The [`MultiheadAttention`](models/attention.py) module implements:
- Parallel attention heads with configurable number of heads
- Scaled dot-product attention
- Linear projections for queries, keys, and values

### Flexible Normalization

The [`ResidualConnection`](utils/residual.py) module supports:
- Standard Layer Normalization ([`NormType.LAYER_NORM`](utils/residual.py))
- Dynamic Tanh normalization ([`NormType.DYT`](utils/residual.py))

## Usage

```python
from models.encoder import TransformerEncoderLayer
from utils.residual import NormType

# Create encoder layer with Layer Normalization
encoder = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    norm_type=NormType.LAYER_NORM
)

# Create encoder layer with DyT normalization
encoder_dyt = TransformerEncoderLayer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    norm_type=NormType.DYT
)

# Forward pass
output = encoder(input_sequence)
```

## Requirements

- PyTorch
- Python 3.6+

## Research Interest

This implementation includes an experimental Dynamic Tanh (DyT) normalization technique as an alternative to standard Layer Normalization, which may be of interest for research into normalization methods in transformer architectures.