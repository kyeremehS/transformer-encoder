import torch
import math
import torch.nn as nn
import torch.nn.functional as F


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