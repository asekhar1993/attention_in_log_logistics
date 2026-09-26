from torch import nn
import torch
import torch.nn.functional as F

class SoftFilterLayer(nn.Module):
    def __init__(self, dim, hidden_size, deep=1, quantile=0.5):
        super().__init__()
        self.quantile = quantile
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        logits = self.layers(x)  # Shape: [B, L, 1]
        
        # Compute adaptive threshold based on quantile
        logits_squeezed = logits.squeeze(-1)  # Shape: [B, L] 
        threshold = torch.quantile(logits_squeezed, self.quantile, dim=1, keepdim=True)  # Shape: [B, 1]
        
        # Generate indices for H_high (task-relevant) and H_low (task-irrelevant)
        idx_high = torch.where(logits_squeezed > threshold)[1]  # Indices where logits > threshold
        idx_low = torch.where(logits_squeezed <= threshold)[1]  # Indices where logits <= threshold
        # Apply filtering (multiply input by logits as in original)
        h = torch.mul(x, logits)  # Shape: [B, L, C]
        
        return h, logits, idx_high, idx_low, threshold