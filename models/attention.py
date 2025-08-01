import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()


        self.d_model = d_model
        # Initialize the weight matrices for queries, keys, and values
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Initialize the output linear layer
        self.W_0 = nn.Linear(d_model, d_model)

    def forward(self, x):
        W = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Compute the attention scores
        attention_scores = torch.matmul(W, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the context vector
        context = torch.matmul(attention_weights, V)

        # Apply the output linear layer
        output = self.W_0(context)

        return output, attention_weights
    

 #Multihead Attention Module
class MultiheadAttention(torch.nn.Module):  
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads

        # Initialize the weight matrices for queries, keys, and values
        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)

        # Initialize the output linear layer
        self.W_0 = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        W_Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        W_K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        W_V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(W_Q, W_K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_weights, W_V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.W_0(context)

        return output, attention_weights
    
    