"""
results/metric_engine.py
========================
Canonical metric engine interface for CA-HTDNet experiments.
Imports and exposes compute_canonical_metrics from results/result_engine.py.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from results.result_engine import compute_canonical_metrics, ENGINE_VERSION

__all__ = ["compute_canonical_metrics", "ENGINE_VERSION"]
