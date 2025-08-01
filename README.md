# Transformer Encoder

A PyTorch implementation of a Transformer encoder with support for both standard Layer Normalization and a novel Dynamic Tanh (DyT) normalization technique.

## Features

- **Multi-Head Attention**: Parallel attention heads for capturing different types of relationships
- **Position-wise Feed Forward Networks**: Two-layer fully connected networks with ReLU activation
- **Flexible Normalization**: Support for both Layer Normalization and Dynamic Tanh (DyT) normalization
- **Positional Encoding**: Sinusoidal positional embeddings for sequence understanding
- **Residual Connections**: Skip connections with dropout for stable training

## Performance Results

### AG News Classification Task

**Layer Normalization Results:**
```
Epoch 1/5 | Train Acc: 60.83% | Val Acc: 79.04%
Epoch 2/5 | Train Acc: 82.86% | Val Acc: 84.39%
Epoch 3/5 | Train Acc: 86.56% | Val Acc: 86.08%
Epoch 4/5 | Train Acc: 88.31% | Val Acc: 87.75%
Epoch 5/5 | Train Acc: 89.58% | Val Acc: 87.70%
```

The model achieves **87.70% validation accuracy** with minimal overfitting and rapid convergence.

## Project Structure

```
transformer-encoder/
├── models/
│   ├── attention.py        # Self-attention and multi-head attention
│   ├── encoder.py          # Transformer encoder and classifier
│   ├── encoder_layer.py    # Individual encoder layer
│   └── feedforward.py      # Position-wise feed forward network
├── utils/
│   ├── DyT.py             # Dynamic Tanh normalization
│   ├── residual.py        # Residual connections
│   ├── postional_encoding.py  # Positional encoding
│   └── data.py            # Data processing utilities
├── train/
│   └── trainer.py         # Training loop and evaluation
└── README.md
```

## Usage

### Basic Training

```python
from train.trainer import train_model

# Train with Layer Normalization
config = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 128,
    "num_classes": 4,
    "use_dyt": False  # Use Layer Normalization
}

train_model(config)
```

### Switch to DyT Normalization

```python
# Train with DyT normalization
config["use_dyt"] = True
train_model(config)
```

## Key Components

### Dynamic Tanh (DyT) Normalization

A novel normalization technique that uses:
- Learnable alpha parameter for controlling tanh saturation
- Element-wise weight and bias parameters
- Tanh activation for bounded outputs

### Multi-Head Attention

- Parallel attention heads with scaled dot-product attention
- Dropout regularization and optional masking support
- Linear projections for queries, keys, and values

## Requirements

```bash
pip install torch torchtext
```

## Research Interest

This implementation provides a foundation for studying normalization techniques in transformer architectures. The experimental Dynamic Tanh (DyT) normalization offers an alternative to standard Layer Normalization

