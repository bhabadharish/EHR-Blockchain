"""
src/models/ca_htdnet_v2.py
==========================
CA-HTDNet V2: Crypto-Agile Healthcare Threat Detection Network (Second Generation)

Architectural Specification:
- Dual-input heterogeneous feature projection (Numeric Branch + Categorical Embeddings)
- Learnable Instance-Level Feature Gating
- Cross-Feature Multi-Head Attention Block
- Dual-Branch Feature Processing:
  - Local Branch: Residual MLP + 1D Temporal/Local Convolution
  - Global Branch: Multi-Head Transformer Encoder
- Gated Residual Fusion:
  - Dynamic local/global balancing + Input projection skip connection
- Deep Representation Refinement Block
- Multi-Task Inference Heads:
  - Attack Classification Head (Class 0: Normal, Class 1: Threat)
  - Calibrated Continuous Threat Risk Score Head in [0.0, 1.0]
"""

import math
from typing import Dict, Any, Tuple, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMLPBlock(nn.Module):
    """Residual Feed-Forward block with LayerNorm and GELU."""
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return self.norm(residual + h)

class CAHTDNetV2(nn.Module):
    """
    Second-generation Crypto-Agile Healthcare Threat Detection Network (CA-HTDNet V2).
    """
    def __init__(
        self,
        num_numerical: int = 18,
        cat_cardinalities: Optional[List[int]] = None,
        d_model: int = 128,
        embedding_dim: int = 16,
        n_heads: int = 4,
        num_transformer_layers: int = 2,
        tcn_kernel_size: int = 3,
        representation_dim: int = 64,
        dropout: float = 0.15,
        num_classes: int = 2,
        # Ablation control flags
        use_gating: bool = True,
        use_attention: bool = True,
        use_residual: bool = True,
        use_tcn: bool = True,
        use_transformer: bool = True,
        use_dual_branch: bool = True
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities or [10, 15, 10]
        self.d_model = d_model
        self.representation_dim = representation_dim
        self.use_gating = use_gating
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_tcn = use_tcn
        self.use_transformer = use_transformer
        self.use_dual_branch = use_dual_branch

        # 1. Numerical Branch Projection
        self.num_proj = nn.Sequential(
            nn.Linear(num_numerical, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Categorical Branch Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=embedding_dim)
            for c in self.cat_cardinalities
        ])
        cat_total_dim = len(self.cat_cardinalities) * embedding_dim
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_total_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 3. Feature Gating Layer
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        else:
            self.gate = None

        # 4. Cross-Feature Multi-Head Attention
        if self.use_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(d_model)
        else:
            self.cross_attn = None

        # 5. Local Branch (Residual MLP + 1D Conv)
        if self.use_dual_branch and self.use_tcn:
            self.local_conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=tcn_kernel_size, padding=tcn_kernel_size // 2),
                nn.BatchNorm1d(16),
                nn.GELU(),
                nn.Conv1d(in_channels=16, out_channels=1, kernel_size=tcn_kernel_size, padding=tcn_kernel_size // 2),
                nn.GELU()
            )
        else:
            self.local_conv = None

        self.local_mlp = ResidualMLPBlock(d_model, expansion=2, dropout=dropout)

        # 6. Global Branch (Transformer Encoder)
        if self.use_dual_branch and self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        else:
            self.transformer = None

        # 7. Gated Residual Fusion
        if self.use_dual_branch:
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        else:
            self.fusion_gate = None

        # 8. Deep Representation Head
        self.rep_block = nn.Sequential(
            nn.Linear(d_model, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 9. Dual Output Heads
        # Attack Head (Classification Logits)
        self.attack_head = nn.Linear(representation_dim, num_classes)
        # Risk Head (Continuous Threat Severity Score [0.0, 1.0])
        self.risk_head = nn.Sequential(
            nn.Linear(representation_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Calibration temperature parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CA-HTDNet V2.
        x_num: (batch_size, num_numerical)
        x_cat: (batch_size, num_categorical)
        """
        # Handle fallback if concatenated
        if x_cat is None and x_num.shape[1] > self.num_numerical:
            x_cat = x_num[:, self.num_numerical:].long()
            x_num = x_num[:, :self.num_numerical].float()
        elif x_cat is None:
            x_cat = torch.zeros(x_num.shape[0], len(self.cat_cardinalities), dtype=torch.long, device=x_num.device)
        else:
            x_cat = x_cat.long()
            x_num = x_num.float()

        # 1. Project Numerical Features
        h_num = self.num_proj(x_num)  # (B, d_model)

        # 2. Embed & Project Categorical Features
        emb_list = []
        for i, emb_layer in enumerate(self.embeddings):
            col_idx = torch.clamp(x_cat[:, i], 0, emb_layer.num_embeddings - 1)
            emb_list.append(emb_layer(col_idx))
        h_cat_raw = torch.cat(emb_list, dim=-1)
        h_cat = self.cat_proj(h_cat_raw)  # (B, d_model)

        # 3. Feature Gating
        if self.use_gating and self.gate is not None:
            g = self.gate(torch.cat([h_num, h_cat], dim=-1))
            h_fused = (g * h_num) + ((1.0 - g) * h_cat)
        else:
            h_fused = (h_num + h_cat) * 0.5

        # 4. Cross-Feature Multi-Head Attention
        if self.use_attention and self.cross_attn is not None:
            # Treat [h_num, h_cat, h_fused] as a 3-token sequence
            token_seq = torch.stack([h_num, h_cat, h_fused], dim=1)  # (B, 3, d_model)
            attn_out, _ = self.cross_attn(token_seq, token_seq, token_seq)
            attn_out = self.attn_norm(token_seq + attn_out)
            h_ctx = attn_out[:, 2, :]  # contextualized fused representation
        else:
            h_ctx = h_fused

        # 5. Local Branch
        h_local = self.local_mlp(h_ctx)
        if self.local_conv is not None:
            # (B, 1, d_model)
            conv_out = self.local_conv(h_local.unsqueeze(1)).squeeze(1)
            h_local = h_local + conv_out

        # 6. Global Branch
        if self.transformer is not None:
            # Transformer over sequence tokens
            seq_in = torch.stack([h_num, h_cat, h_local], dim=1)  # (B, 3, d_model)
            trans_out = self.transformer(seq_in)
            h_global = trans_out[:, 2, :]
        else:
            h_global = h_local

        # 7. Gated Residual Fusion
        if self.fusion_gate is not None:
            alpha = self.fusion_gate(torch.cat([h_local, h_global], dim=-1))
            h_combined = (alpha * h_local) + ((1.0 - alpha) * h_global)
        else:
            h_combined = h_local

        # Residual connection from input fusion
        if self.use_residual:
            h_combined = h_combined + h_ctx

        # 8. Representation Head
        rep = self.rep_block(h_combined)  # (B, representation_dim)

        # 9. Multi-Task Heads
        logits = self.attack_head(rep)
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.01)
        probabilities = F.softmax(calibrated_logits, dim=-1)
        risk_score = self.risk_head(rep)

        return {
            "logits": logits,
            "calibrated_logits": calibrated_logits,
            "probabilities": probabilities,
            "risk_score": risk_score,
            "representation": rep
        }
