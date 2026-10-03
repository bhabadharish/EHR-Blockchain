"""
src/models/ca_htdnet.py
=======================
CA-HTDNet: Crypto-Agile Healthcare Threat Detection Network
Canonical proposed architecture adhering to Master Specification:

                 INPUT SECURITY FEATURES
                         │
              ┌──────────┴──────────┐
              │                     │
        NUMERIC BRANCH        CATEGORICAL BRANCH
              │                     │
        Normalization          Embeddings
              │                     │
        Feature Projection      Feature Projection
              │                     │
              └──────────┬──────────┘
                         │
                  FEATURE FUSION
                         │
              ┌──────────┴──────────┐
              │                     │
        Local Interaction      Global Interaction
              │                     │
           TCN (1D Conv)       Multi-Head Attention
              │                     │
              └──────────┬──────────┘
                         │
                 GATED FUSION
                         │
                   Dense Layers (Residual MLP)
                         │
              ┌──────────┴──────────┐
              │                     │
         Attack Head           Risk Head
              │                     │
         Class Probability     Risk Score
              │                     │
              └──────────┬──────────┘
                         │
                 Threat Decision
"""

import math
from typing import Dict, Any, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn import TemporalConvNet

class FeatureTokenizer(nn.Module):
    """
    Projects continuous numerical features into dense feature tokens with learned gating.
    """
    def __init__(self, num_features: int, d_model: int, use_gating: bool = True):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.use_gating = use_gating
        
        self.weights = nn.Parameter(torch.randn(num_features, d_model) / math.sqrt(d_model))
        self.biases = nn.Parameter(torch.zeros(num_features, d_model))
        self.norm = nn.LayerNorm(d_model)
        
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_features)
        # tokens: (batch_size, num_features, d_model)
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        tokens = self.norm(tokens)
        if self.use_gating:
            g = self.gate(tokens)
            tokens = tokens * g
        return tokens

class CategoricalTokenizer(nn.Module):
    """
    Embeds categorical features and projects into d_model dimension.
    """
    def __init__(self, cardinalities: list, d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=d_model)
            for c in cardinalities
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        # x_cat: (batch_size, num_cat)
        tokens = []
        for i, emb in enumerate(self.embeddings):
            col = x_cat[:, i]
            # clamp indices within valid embedding vocabulary
            col = torch.clamp(col, 0, emb.num_embeddings - 1)
            tokens.append(emb(col).unsqueeze(1))
        # (batch_size, num_cat, d_model)
        cat_tokens = torch.cat(tokens, dim=1)
        return self.norm(cat_tokens)

class CA_HTDNet(nn.Module):
    """
    Full CA-HTDNet Architecture with modular ablation switches:
      - use_residual: Residual connections in dense blocks
      - use_gating: Learned feature gating
      - use_attention: Multi-head self-attention
      - use_tcn: 1D causal dilated convolutions
    """
    def __init__(
        self,
        num_numerical: int = 13,
        num_categories: int = 3,
        cardinalities: Optional[list] = None,
        d_model: int = 64,
        tcn_channels: list = [32, 64],
        gru_hidden: int = 64,
        attention_heads: int = 4,
        attention_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.10,
        use_residual: bool = True,
        use_gating: bool = True,
        use_attention: bool = True,
        use_tcn: bool = True
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categories = num_categories
        self.d_model = d_model
        self.use_residual = use_residual
        self.use_gating = use_gating
        self.use_attention = use_attention
        self.use_tcn = use_tcn
        
        cardinalities = cardinalities or [16, 20, 16]

        # 1. Numeric Branch Tokenizer
        self.num_tokenizer = FeatureTokenizer(num_numerical, d_model, use_gating=use_gating)

        # 2. Categorical Branch Tokenizer
        self.cat_tokenizer = CategoricalTokenizer(cardinalities, d_model)

        total_features = num_numerical + num_categories
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Global Interaction: Multi-Head Self-Attention
        if self.use_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=attention_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=attention_layers,
                enable_nested_tensor=False
            )
            self.attn_norm = nn.LayerNorm(d_model)
        else:
            self.transformer = None

        # 4. Local Interaction: TCN (1D Dilated Convolutions)
        if self.use_tcn:
            self.tcn = TemporalConvNet(
                num_inputs=d_model,
                num_channels=tcn_channels,
                kernel_size=3,
                dropout=dropout
            )
            self.tcn_pool = nn.AdaptiveAvgPool1d(1)
            self.tcn_proj = nn.Linear(tcn_channels[-1], d_model)
        else:
            self.tcn = None

        # 5. Gated Fusion Mechanism
        # Dynamically balances Local vs Global representations
        if self.use_attention and self.use_tcn:
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        elif self.use_gating:
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
        else:
            self.fusion_gate = None

        # 6. Dense Layers
        self.dense_1 = nn.Linear(d_model, 128)
        self.norm_1 = nn.LayerNorm(128)
        self.dense_2 = nn.Linear(128, 64)
        self.norm_2 = nn.LayerNorm(64)
        self.res_proj = nn.Linear(d_model, 64) if self.use_residual else None
        self.dropout = nn.Dropout(dropout)

        # 7. Dual Heads: Attack Head (Classification) & Risk Head (Threat Score)
        self.attack_head = nn.Linear(64, num_classes)
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Calibrated temperature scaling parameter (frozen during training, learned during calibration)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Handle concatenated inputs (N, 16)
        if x_cat is None and x_num.shape[1] > self.num_numerical:
            x_cat = x_num[:, self.num_numerical:].long()
            x_num = x_num[:, :self.num_numerical].float()
        elif x_cat is None:
            # Fallback zero categories
            x_cat = torch.zeros(x_num.shape[0], self.num_categories, dtype=torch.long, device=x_num.device)
        else:
            x_cat = x_cat.long()
            x_num = x_num.float()

        batch_size = x_num.shape[0]

        # 1. Feature Tokenization (Numeric + Categorical)
        num_tokens = self.num_tokenizer(x_num) # (batch, num_numerical, d_model)
        cat_tokens = self.cat_tokenizer(x_cat) # (batch, num_categories, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 2. Feature Fusion
        tokens = torch.cat([cls_tokens, num_tokens, cat_tokens], dim=1) # (batch, 1 + total_feat, d_model)

        # 3. Global Interaction (Attention)
        if self.use_attention:
            attn_out = self.transformer(tokens)
            attn_rep = self.attn_norm(attn_out[:, 0, :]) # CLS token representation
        else:
            attn_rep = tokens.mean(dim=1)

        # 4. Local Interaction (TCN)
        if self.use_tcn:
            # TCN expects (batch, channels, seq_len)
            tcn_in = tokens.permute(0, 2, 1)
            tcn_out = self.tcn(tcn_in)
            tcn_pooled = self.tcn_pool(tcn_out).squeeze(-1) # (batch, tcn_channels[-1])
            local_rep = F.gelu(self.tcn_proj(tcn_pooled)) # (batch, d_model)
        else:
            local_rep = tokens.mean(dim=1)

        # 5. Gated Fusion
        if self.use_attention and self.use_tcn:
            gate_input = torch.cat([local_rep, attn_rep], dim=-1)
            gate_weight = self.fusion_gate(gate_input)
            fused = gate_weight * local_rep + (1.0 - gate_weight) * attn_rep
        elif self.fusion_gate is not None:
            fused = self.fusion_gate(attn_rep) * attn_rep
        else:
            fused = 0.5 * (local_rep + attn_rep)

        # 6. Dense Layers with Residual Connection
        h = F.gelu(self.norm_1(self.dense_1(fused)))
        h = self.dropout(h)
        h = self.norm_2(self.dense_2(h))
        if self.use_residual and self.res_proj is not None:
            h = h + self.res_proj(fused)
        h = F.gelu(h)
        h = self.dropout(h)

        # 7. Output Heads
        raw_logits = self.attack_head(h)
        calibrated_logits = raw_logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        risk_score = self.risk_head(h).squeeze(-1)

        tri_state_logits = torch.stack([
            raw_logits[:, 0],
            0.5 * (raw_logits[:, 0] + raw_logits[:, 1]),
            raw_logits[:, 1]
        ], dim=-1)

        return {
            "logits": raw_logits,
            "calibrated_logits": calibrated_logits,
            "tri_state_logits": tri_state_logits,
            "risk_score": risk_score,
            "latent": h,
            "fused": fused
        }

def build_ablation_model(variant: str, num_numerical: int = 13, num_categories: int = 3) -> CA_HTDNet:
    """
    Factory creating exact ablation variants:
      A0: Base MLP
      A1: + Residual Connections
      A2: + Feature Gating
      A3: + Attention
      A4: + Focal Loss (handled at loss level)
      A5: + Class Weighting (handled at loss level)
      A6: + TCN
      A7: + Attention + Gating
      A8: Full CA-HTDNet
    """
    v = variant.upper()
    if v == "A0":
        return CA_HTDNet(num_numerical, num_categories, use_residual=False, use_gating=False, use_attention=False, use_tcn=False)
    elif v == "A1":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=False, use_attention=False, use_tcn=False)
    elif v == "A2":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=True, use_attention=False, use_tcn=False)
    elif v == "A3":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=False, use_attention=True, use_tcn=False)
    elif v in ["A4", "A5"]:
        # A4 (+ Focal Loss) and A5 (+ Class Weighting) use A3 architecture with loss modifications
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=False, use_attention=True, use_tcn=False)
    elif v == "A6":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=False, use_attention=False, use_tcn=True)
    elif v == "A7":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=True, use_attention=True, use_tcn=False)
    elif v == "A8":
        return CA_HTDNet(num_numerical, num_categories, use_residual=True, use_gating=True, use_attention=True, use_tcn=True)
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
