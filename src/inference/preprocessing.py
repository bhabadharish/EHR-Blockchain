"""
src/inference/preprocessing.py
==============================
Offline-fitted preprocessing wrapper for real-time inference.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union, List

from src.data.feature_pipeline_v2 import FeaturePipelineV2

class InferencePreprocessor:
    def __init__(self, preprocessor_path: str = "models/preprocessors/preprocessor.pkl"):
        if not os.path.exists(preprocessor_path):
            preprocessor_path = "experiments/CAHTDNET_V2_001/preprocessing/preprocessor.pkl"
        self.pipeline = FeaturePipelineV2.load(preprocessor_path)

    def transform_single(self, event: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame([event])
        return self.pipeline.transform(df)

    def transform_batch(self, events: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.DataFrame(events)
        return self.pipeline.transform(df)
