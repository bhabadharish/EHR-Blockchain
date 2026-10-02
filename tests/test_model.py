"""
tests/test_model.py
Tests for neural network architectures, loss functions, and gradient computations (Phases 15 & 17).
"""

import pytest
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models.architectures import (
    ProposedLightweightTCNConvAttention,
    DepthwiseSeparableCNNModel,
    LightweightTCNAttentionModel,
    LightweightTCNModel,
    ProposedTCNTransformerAttention, TCNOnlyModel, TransformerOnlyModel,
    TCNTransformerModel, MLPModel, CNN1DModel, BiLSTMModel
)
from backend.models.loss import MultiClassFocalLoss, compute_class_weights
from backend.models.baselines import get_baseline_model

@pytest.fixture
def dummy_batch():
    batch_size = 16
    in_features = 50
    return torch.randn(batch_size, in_features)

@pytest.mark.parametrize("model_cls", [
    ProposedLightweightTCNConvAttention,
    DepthwiseSeparableCNNModel,
    LightweightTCNAttentionModel,
    LightweightTCNModel,
    ProposedTCNTransformerAttention,
    TCNOnlyModel,
    TransformerOnlyModel,
    TCNTransformerModel,
    MLPModel,
    CNN1DModel,
    BiLSTMModel
])
def test_model_forward_and_backward(model_cls, dummy_batch):
    num_classes = 15
    model = model_cls(in_features=50, num_classes=num_classes)
    logits = model(dummy_batch)
    assert logits.shape == (16, num_classes)

    # Backward pass with loss
    targets = torch.randint(0, num_classes, (16,))
    loss_fn = MultiClassFocalLoss()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0

    loss.backward()
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad is True

def test_class_weights_computation():
    y = np.array([0, 0, 0, 1, 2, 2, 3])
    weights = compute_class_weights(y, num_classes=4)
    assert len(weights) == 4
    # Rare class 1 should have higher weight than frequent class 0
    assert weights[1] > weights[0]

def test_classical_baselines_instantiation():
    lr = get_baseline_model("LR")
    rf = get_baseline_model("RF")
    lgbm = get_baseline_model("LGBM")
    assert lr is not None
    assert rf is not None
    assert lgbm is not None
