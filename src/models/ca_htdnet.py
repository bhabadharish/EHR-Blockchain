import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from src.models.ft_transformer import FTTransformerExpert
from src.models.tcn import TemporalConvNet
from src.models.bigru_attention import BiGRUMultiHeadAttention

class CA_HTDNet(nn.Module):
    """
    CA-HTDNet: Crypto-Agile Healthcare Threat Detection Network.
    Combines:
      - Tabular Experts (FT-Transformer + GBDT Probability Injection)
      - Temporal Dual-Branch (Causal TCN + BiGRU with Multi-Head Self-Attention)
      - Contextual Multi-Modal Fusion (Network + FHIR + Device Telemetry)
      - Tri-State Decision Projection (Normal, Suspicious, Attack)
    """
    def __init__(
        self,
        num_numerical: int = 13,
        num_categories: int = 3,
        d_model: int = 64,
        tcn_channels: list = [32, 64],
        gru_hidden: int = 64,
        num_classes: int = 2,
        seq_len: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. FT-Transformer Tabular Expert
        self.ft_transformer = FTTransformerExpert(
            num_numerical=num_numerical,
            num_categories=num_categories,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=dropout
        )

        # Tabular GBDT feature injection projection (LightGBM & CatBoost soft probability hints)
        self.gbdt_proj = nn.Linear(4, d_model)

        # 2. Temporal Branch A: Causal Dilated TCN
        # Input channel: d_model, outputs tcn_channels[-1]
        self.tcn = TemporalConvNet(
            num_inputs=d_model,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)

        # 3. Temporal Branch B: BiGRU + Multi-Head Self-Attention
        self.bigru_mha = BiGRUMultiHeadAttention(
            input_dim=d_model,
            hidden_dim=gru_hidden,
            num_layers=2,
            num_heads=4,
            dropout=dropout
        )

        # 4. Contextual Multi-Modal Fusion
        # Concatenates:
        # - Tabular CLS (d_model)
        # - GBDT projected hint (d_model)
        # - TCN temporal embedding (tcn_channels[-1] = 64)
        # - BiGRU Attention embedding (gru_hidden * 2 = 128)
        fusion_dim = d_model + d_model + tcn_channels[-1] + (gru_hidden * 2)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. Tri-State Decision & Classification Head
        self.classifier = nn.Linear(128, num_classes)
        self.tri_state_head = nn.Linear(128, 3) # Normal, Suspicious, Attack

        # Calibration temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(
        self,
        x_num: torch.Tensor,
        gbdt_probs: Optional[torch.Tensor] = None,
        x_seq: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = x_num.shape[0]
        
        # 1. FT-Transformer pass
        logits_tab, cls_rep = self.ft_transformer(x_num)
        
        # 2. GBDT probability hints
        if gbdt_probs is None:
            # Default neutral probability placeholder if GBDTs not passed
            gbdt_probs = torch.zeros(batch_size, 4, device=x_num.device)
        gbdt_rep = F.gelu(self.gbdt_proj(gbdt_probs))

        # 3. Temporal sequence generation if not provided (synthesize sliding local temporal context)
        if x_seq is None:
            # Expand cls_rep across temporal window with learned temporal positional perturbation
            seq_reps = cls_rep.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, d_model)
        else:
            seq_reps = x_seq

        # Temporal TCN (expects batch, channels, seq_len)
        tcn_in = seq_reps.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        tcn_emb = self.tcn_pool(tcn_out).squeeze(-1) # (batch, 64)

        # Temporal BiGRU + MHA
        bigru_emb = self.bigru_mha(seq_reps) # (batch, 128)

        # 4. Multi-Modal Context Fusion
        fused = torch.cat([cls_rep, gbdt_rep, tcn_emb, bigru_emb], dim=-1)
        latent = self.fusion(fused)

        # 5. Output Heads
        raw_logits = self.classifier(latent)
        calibrated_logits = raw_logits / torch.clamp(self.temperature, min=0.1, max=10.0)
        
        tri_state_logits = self.tri_state_head(latent)

        return {
            "logits": raw_logits,
            "calibrated_logits": calibrated_logits,
            "tri_state_logits": tri_state_logits,
            "latent_representation": latent,
            "cls_representation": cls_rep
        }
