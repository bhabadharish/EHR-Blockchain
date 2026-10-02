"""
backend/models/architectures.py
Deep learning model architectures for intelligent cyber threat detection (Phase 15)
and ablation studies (Phase 21).
Includes Proposed TCN-Transformer-Attention, Baselines (MLP, 1D-CNN, BiLSTM), and Ablations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock1D(nn.Module):
    """Temporal Convolutional Network residual block with dilated causal 1D convolutions."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        res = self.downsample(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res

class DepthwiseSeparableConv1d(nn.Module):
    """Lightweight 1D Depthwise-Separable Convolution."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class ChannelAttention1D(nn.Module):
    """Squeeze-and-Excitation channel attention mechanism for 1D representations."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        w = self.fc(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * w

# Proposed Architecture: Lightweight TCN + Separable 1D Conv + Channel Attention (Apple M2 optimized)
class ProposedLightweightTCNConvAttention(nn.Module):
    """
    Primary Proposed Model Architecture:
    Input features -> Dense projection (32) -> Depthwise-Separable Conv1D ->
    TCN Block 1 -> TCN Block 2 (Residual) -> Lightweight Channel Attention ->
    Global Average Pooling -> Dense(32) -> Dropout(0.1-0.2) -> Classification.
    Parameters: < 20,000 (well within the < 1M - 2M Apple M2 budget).
    """
    def __init__(
        self,
        in_features: int = 50,
        num_classes: int = 15,
        hidden_dim: int = 32,
        tcn_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. Feature Preprocessing / Dense Projection: 32 dimensions
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # 2. Depthwise-Separable Conv1D
        self.sep_conv = DepthwiseSeparableConv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # 3. Temporal Convolutional Network blocks with residual connections
        self.tcn_block1 = TCNBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout)
        self.tcn_block2 = TCNBlock1D(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=2, dropout=dropout)
        
        # 4. Lightweight Channel Attention (Squeeze-and-Excitation)
        self.channel_attn = ChannelAttention1D(channels=hidden_dim, reduction=4)

        # 5. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 6. Small Dense Layer & Dropout
        self.dense_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 7. Output Classification
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        # Dense projection to 32 dimensions
        proj = self.proj(x)  # (batch, hidden_dim)
        
        # Reshape to sequence for 1D convolutions: (batch, channels=1, length=hidden_dim)
        seq = proj.unsqueeze(1)
        
        # Depthwise-Separable Conv1D: (batch, hidden_dim, length=hidden_dim)
        out = self.sep_conv(seq)
        
        # TCN Residual Blocks
        out = self.tcn_block1(out)
        out = self.tcn_block2(out)
        
        # Lightweight Channel Attention
        out = self.channel_attn(out)
        
        # Global Average Pooling: (batch, hidden_dim)
        pooled = self.gap(out).squeeze(-1)
        
        # Small Dense Layer & Dropout
        feat = self.dense_head(pooled)
        
        # Classification logits
        return self.classifier(feat)

# Baseline 9: Depthwise-Separable CNN
class DepthwiseSeparableCNNModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim)
        self.sep_conv1 = DepthwiseSeparableConv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.sep_conv2 = DepthwiseSeparableConv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.proj(x).unsqueeze(1)
        out = self.sep_conv2(self.sep_conv1(proj))
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)

# Baseline 8: Lightweight TCN + Attention
class LightweightTCNAttentionModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(in_features, channels)
        self.tcn1 = TCNBlock1D(1, channels, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock1D(channels, channels, kernel_size=3, dilation=2, dropout=dropout)
        self.attn = ChannelAttention1D(channels, reduction=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.proj(x).unsqueeze(1)
        out = self.tcn2(self.tcn1(seq))
        out = self.attn(out)
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)

# Baseline 7: Lightweight TCN
class LightweightTCNModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, channels: int = 32, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(in_features, channels)
        self.tcn1 = TCNBlock1D(1, channels, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock1D(channels, channels, kernel_size=3, dilation=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.proj(x).unsqueeze(1)
        out = self.tcn2(self.tcn1(seq))
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)

# Legacy / Heavy Transformer Ablation Models (Retained for research ablation comparison)
class ProposedTCNTransformerAttention(nn.Module):
    def __init__(
        self,
        in_features: int = 50,
        num_classes: int = 15,
        tcn_channels: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Feature Projection & Shaping: (batch, in_features) -> (batch, 1, in_features) -> TCN
        self.input_proj = nn.Linear(in_features, in_features)
        
        # 1. Temporal Convolutional Network (TCN) with multi-dilation receptive field
        self.tcn_block1 = TCNBlock1D(1, tcn_channels, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn_block2 = TCNBlock1D(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn_block3 = TCNBlock1D(tcn_channels, d_model, kernel_size=3, dilation=4, dropout=dropout)

        # 2. Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 3. Multi-Head Attention Layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # 4. Global Temporal Pooling & Classification Head
        self.fc1 = nn.Linear(d_model * 2, 64)
        self.act_fc = nn.GELU()
        self.fc_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        x_proj = self.input_proj(x)
        x_seq = x_proj.unsqueeze(1)  # (batch, 1, in_features)

        # TCN Feature Extraction
        tcn_out = self.tcn_block1(x_seq)
        tcn_out = self.tcn_block2(tcn_out)
        tcn_out = self.tcn_block3(tcn_out)  # (batch, d_model, in_features)

        # Reshape for Transformer: (batch, seq_len=in_features, d_model)
        trans_in = tcn_out.transpose(1, 2)

        # Transformer Contextual Modeling
        trans_out = self.transformer_encoder(trans_in)

        # Multi-Head Attention Attentive Pooling
        attn_out, _ = self.multihead_attn(trans_out, trans_out, trans_out)
        attn_out = self.attn_norm(trans_out + attn_out)

        # Global Avg + Max Temporal Pooling
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, d_model * 2)

        # Dense Classification Head
        feat = self.fc_drop(self.act_fc(self.fc1(pooled)))
        logits = self.classifier(feat)
        return logits

# Ablation A: TCN only
class TCNOnlyModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, channels: int = 64, dropout: float = 0.15):
        super().__init__()
        self.tcn1 = TCNBlock1D(1, channels, kernel_size=3, dilation=1, dropout=dropout)
        self.tcn2 = TCNBlock1D(channels, channels, kernel_size=3, dilation=2, dropout=dropout)
        self.tcn3 = TCNBlock1D(channels, channels, kernel_size=3, dilation=4, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tcn3(self.tcn2(self.tcn1(x.unsqueeze(1))))
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)

# Ablation B: Transformer only
class TransformerOnlyModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, d_model: int = 64, nhead: int = 4, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x.unsqueeze(-1))  # (batch, in_features, d_model)
        out = self.transformer(tokens)
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)

# Ablation C: TCN + Transformer (without extra attention layer)
class TCNTransformerModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, d_model: int = 64, nhead: int = 4, dropout: float = 0.15):
        super().__init__()
        self.tcn = TCNBlock1D(1, d_model, kernel_size=3, dilation=2, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tcn_out = self.tcn(x.unsqueeze(1)).transpose(1, 2)
        trans_out = self.transformer(tcn_out)
        pooled = torch.mean(trans_out, dim=1)
        return self.fc(pooled)

# Baseline 4: Multi-Layer Perceptron (MLP)
class MLPModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Baseline 5: 1D-CNN
class CNN1DModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.fc(feat)

# Baseline 6: BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, in_features: int = 50, num_classes: int = 15, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1)  # (batch, seq_len=in_features, 1)
        out, _ = self.lstm(tokens)
        pooled = torch.mean(out, dim=1)
        return self.fc(pooled)
