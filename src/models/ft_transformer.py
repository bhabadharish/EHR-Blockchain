import math
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericalFeatureTokenizer(nn.Module):
    """
    Transforms continuous numerical features into dense d_model embedding tokens.
    """
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, d_model) / math.sqrt(d_model))
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_features)
        # tokens: (batch_size, num_features, d_model)
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return self.norm(tokens)

class FTTransformerExpert(nn.Module):
    """
    Feature Tokenizer Transformer (FT-Transformer) Expert for Tabular Telemetry.
    """
    def __init__(
        self,
        num_numerical: int,
        num_categories: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(num_numerical, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2)
        )

    def forward(self, x_num: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_num: (batch, num_features)
        batch_size = x_num.shape[0]
        tokens = self.tokenizer(x_num) # (batch, num_features, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1) # (batch, 1 + num_features, d_model)
        
        trans_out = self.transformer(x_seq)
        cls_rep = trans_out[:, 0, :] # Extract CLS embedding (batch, d_model)
        logits = self.head(cls_rep)
        return logits, cls_rep
