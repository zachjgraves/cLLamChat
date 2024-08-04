
# Import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from zeta.nn.modules.p_scan import pscan


# Define Mamba block
class MambaBlock(nn.Module):
    def __init__(self, n_features, n_states, expansion=1):
        super(MambaBlock, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_states = n_states
        self.expansion = expansion

        # Calculate constants
        n_features_inner = int(n_features * expansion)
        n_delta = math.ceil(n_features / 16)
        self.n_features_inner = n_features_inner
        self.n_delta = n_delta

        # Initialize projection blocks
        self.project_in = nn.Sequential(
            nn.GroupNorm(1, n_features, affine=False),  # Normalize input
            nn.Linear(n_features, n_features_inner),
        )
        self.project_out = nn.Sequential(
            nn.InstanceNorm1d(n_features_inner),
            nn.Linear(n_features_inner, n_features),
        )

        # Initialize convolution block
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                n_features_inner,
                n_features_inner,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm1d(n_features_inner),
            nn.SiLU(),
        )

        # Initialize tensors
        A = torch.arange(1, n_states + 1).unsqueeze(0).repeat(n_features_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(n_features_inner))

        # Initialize get tensor blocks
        self.get_C = nn.Linear(n_features_inner, n_states)
        self.get_B = nn.Linear(n_features_inner, n_states)
        self.get_Delta = nn.Sequential(
            nn.Linear(n_features_inner, n_delta),
            nn.Linear(n_delta, n_features_inner),
            nn.Softplus(),
        )

    def forward(self, x):

        # Save original input for skip connection
        x0 = x

        ### START MAMBA ###

        # Project into inner dimension
        h = self.project_in(x)
        h0 = h  # Save for residual connection

        # Apply convolution over sequence
        h = rearrange(h, 'b l f -> b f l')
        h = self.conv_block(h)
        h = rearrange(h, 'b f l -> b l f')

        # Apply SSM
        h = self.ssm(h)

        # Apply residual connection
        h = h * F.silu(h0)

        # Project back
        x = self.project_out(h)

        # Apply residual connection
        x = x + x0

        # Return x
        return x

    
    def ssm(self, h):

        # Get tensors
        A = torch.exp(self.A_log)  # (n_features_inner, n_states)
        B = self.get_B(h)          # (B, L, n_states)
        C = self.get_C(h)          # (B, L, n_states)
        D = self.D                 # (n_features_inner)
        Delta = self.get_Delta(h)  # (B, L, n_features_inner)

        # Perform parallel scan
        # y = selective_scan(h, Delta, A, B, C, D)
        
        # Calulate tensors
        A_bar = torch.exp(-Delta.unsqueeze(-1) * A)  # (B, L, ED, N)
        B_bar = Delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, ED, N)
        BX = B_bar * h.unsqueeze(-1)  # (B, L, ED, N)

        # Get states through parallel scan
        states = pscan(A_bar, BX)

        # Get output
        y = (states @ C.unsqueeze(-1)).squeeze()
        y = y + D * h
            
        # Return y
        return y

# Test
if __name__ == "__main__":

    # Create random tensor
    x = torch.rand(32, 10, 512)

    # Create Mamba block
    mamba_block = MambaBlock(512, 16)

    # Test Mamba block
    y = mamba_block(x)

    # Done
    print("Done")

    