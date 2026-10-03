"""
src/models/ft_transformer.py
============================
Feature Tokenizer Transformer (FT-Transformer) Tabular Baseline
"""

import math
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_features, d_model) / math.sqrt(d_model))
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return self.norm(tokens)

class FTTransformerExpert(nn.Module):
    def __init__(
        self,
        num_numerical: int = 13,
        num_categories: int = 3,
        cardinalities: Optional[list] = None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.10,
        num_classes: int = 2
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        cardinalities = cardinalities or [16, 20, 16]

        self.num_tokenizer = NumericalFeatureTokenizer(num_numerical, d_model)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(c, d_model) for c in cardinalities
        ])
        self.cat_norm = nn.LayerNorm(d_model)
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
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_cat is None and x_num.shape[1] > self.num_numerical:
            x_cat = x_num[:, self.num_numerical:].long()
            x_num = x_num[:, :self.num_numerical].float()
        elif x_cat is None:
            x_cat = torch.zeros(x_num.shape[0], self.num_categories, dtype=torch.long, device=x_num.device)
        else:
            x_cat = x_cat.long()
            x_num = x_num.float()

        batch_size = x_num.shape[0]
        num_toks = self.num_tokenizer(x_num)
        
        cat_toks = []
        for i, emb in enumerate(self.cat_embeddings):
            col = torch.clamp(x_cat[:, i], 0, emb.num_embeddings - 1)
            cat_toks.append(emb(col).unsqueeze(1))
        cat_toks = self.cat_norm(torch.cat(cat_toks, dim=1))

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, num_toks, cat_toks], dim=1)
        
        trans_out = self.transformer(seq)
        cls_rep = trans_out[:, 0, :]
        logits = self.head(cls_rep)
        return logits, cls_rep
