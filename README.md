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

The model demonstrates excellent learning progression with:
- **Strong validation performance**: 87.70% final accuracy
- **Good generalization**: Minimal overfitting (train: 89.58% vs val: 87.70%)
- **Rapid convergence**: Significant improvement from epoch 1 to 2

## Architecture

The transformer encoder consists of the following key components:

### Core Models
- [`MultiheadAttention`](models/attention.py) - Multi-head self-attention mechanism
- [`SelfAttention`](models/attention.py) - Single-head self-attention implementation
- [`PositionWiseFeed`](models/feedforward.py) - Position-wise feed forward network
- [`TransformerEncoderLayer`](models/encoder_layer.py) - Complete encoder layer combining attention and feed forward

### Utilities
- [`DyT`](utils/DyT.py) - Dynamic Tanh normalization layer (novel alternative to Layer Norm)
- [`ResidualConnection`](utils/residual.py) - Residual connections with configurable normalization
- [`PositionalEncoding`](utils/postional_encoding.py) - Sinusoidal positional embeddings
- [`DataLoader`](utils/data.py) - AG News dataset processing and batching

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

## Key Components

### Dynamic Tanh (DyT) Normalization

A novel normalization technique implemented in [`DyT`](utils/DyT.py) that uses:
- Learnable alpha parameter for controlling tanh saturation
- Element-wise weight and bias parameters
- Tanh activation for bounded outputs

**Research Potential**: Compare DyT vs Layer Norm performance on various tasks

### Multi-Head Attention

The [`MultiheadAttention`](models/attention.py) module implements:
- Parallel attention heads with configurable number of heads
- Scaled dot-product attention with dropout
- Linear projections for queries, keys, and values
- Optional attention mask support

### Flexible Normalization

The [`ResidualConnection`](utils/residual.py) module supports:
- Standard Layer Normalization ([`NormType.LAYER_NORM`](utils/residual.py))
- Dynamic Tanh normalization ([`NormType.DYT`](utils/residual.py))

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

### Model Configuration

```python
from models.encoder import TransformerClassifier
from utils.residual import NormType

# Create model with Layer Normalization
config_ln = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 128,
    "num_classes": 4,
    "use_dyt": False
}

model_ln = TransformerClassifier(vocab_size=50000, config=config_ln)

# Create model with DyT normalization
config_dyt = config_ln.copy()
config_dyt["use_dyt"] = True

model_dyt = TransformerClassifier(vocab_size=50000, config=config_dyt)
```

### Custom Encoder Layer

```python
from models.encoder_layer import TransformerEncoderLayer
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

## Experimental Setup

### Dataset
- **AG News**: 4-class text classification (World, Sports, Business, Technology)
- **Preprocessing**: Basic English tokenization with padding to max_len=128
- **Vocabulary**: Built from training data with `<pad>` and `<unk>` tokens

### Model Configuration
```python
config = {
    "d_model": 256,
    "num_heads": 8, 
    "num_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_len": 128,
    "num_classes": 4,
    "batch_size": 64
}
```

### Training Details
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: CrossEntropyLoss
- **Epochs**: 5
- **Validation**: Evaluated each epoch

## Requirements

```
torch>=1.9.0
torchtext>=0.10.0
python>=3.7
```

Install dependencies:
```bash
pip install torch torchtext
```

## Future Work

1. **Normalization Comparison**: Systematic comparison between Layer Norm and DyT normalization across multiple datasets
2. **Attention Analysis**: Visualization of attention patterns and head specialization
3. **Scaling Studies**: Performance evaluation with different model sizes
4. **Domain Adaptation**: Testing on different text classification tasks

## Research Interest

This implementation provides a clean foundation for studying:
- **Normalization techniques** in transformer architectures
- **Attention mechanisms** and their interpretability  
- **Architectural variations** and their impact on performance

The experimental Dynamic Tanh (DyT) normalization offers an interesting alternative to standard Layer Normalization for research into normalization methods in transformer architectures.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{transformer_encoder_dyt,
  title={Transformer Encoder with Dynamic Tanh Normalization},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/transformer-encoder}
}
```