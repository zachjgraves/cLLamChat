
# Import libraries
import math
import torch
import torch.nn as nn


# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=8, expansion=1):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.ReLU(),
            nn.Linear(n_features_inner, n_features),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):

        # Apply self-attention
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        return x
    
# Test
if __name__ == '__main__':

    # Create random tensor
    x = torch.rand(32, 10, 512)

    # Create transformer block
    transformer_block = TransformerBlock(512)

    # Test transformer block
    y = transformer_block(x)

    # Done
    print('Done')

