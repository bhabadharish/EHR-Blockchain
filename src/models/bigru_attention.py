import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUMultiHeadAttention(nn.Module):
    """
    Bidirectional GRU with Multi-Head Self-Attention for Temporal Sequence Modeling.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.gru_dim = hidden_dim * 2 # Bidirectional
        
        self.mha = nn.MultiheadAttention(
            embed_dim=self.gru_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.bigru(x) # (batch, seq_len, gru_dim)
        
        # Self-attention over temporal sequence
        attn_out, _ = self.mha(gru_out, gru_out, gru_out)
        norm_out = self.layer_norm(gru_out + self.dropout(attn_out))
        
        # Temporal pooling (average over sequence)
        seq_rep = torch.mean(norm_out, dim=1) # (batch, gru_dim)
        return seq_rep
