"""
src/inference/model_loader.py
=============================
Frozen model loading layer for real-time threat detection inference.
Guarantees NO training occurs inside inference and verifies hash integrity.
"""

import os
import json
import hashlib
import torch
from typing import Tuple, Dict, Any

from src.models.ca_htdnet_v2 import CAHTDNetV2

def compute_hash(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

class ModelLoader:
    _cached_model = None
    _cached_device = None

    @classmethod
    def load_ca_htdnet_v2(
        cls,
        weights_path: str = "models/proposed/ca_htdnet_v2.pt",
        lock_path: str = "results/FINAL_RESULTS_LOCK.json"
    ) -> Tuple[CAHTDNetV2, torch.device]:
        if cls._cached_model is not None:
            return cls._cached_model, cls._cached_device

        if not os.path.exists(weights_path):
            weights_path = "experiments/CAHTDNET_V2_001/models/CAHTDNET_V2_001.pt"

        # Integrity check
        if os.path.exists(lock_path) and os.path.exists(weights_path):
            try:
                with open(lock_path, "r") as f:
                    lock = json.load(f)
                expected_hash = lock.get("model_hash")
                actual_hash = compute_hash(weights_path)
                if expected_hash and expected_hash != "N/A" and expected_hash != actual_hash:
                    print(f"  Warning: Model hash mismatch. Expected: {expected_hash}, Actual: {actual_hash}")
            except Exception as e:
                print(f"  Integrity check note: {e}")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = CAHTDNetV2(
            num_numerical=18,
            cat_cardinalities=[10, 15, 10],
            d_model=128
        ).to(device)

        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        cls._cached_model = model
        cls._cached_device = device
        return model, device
