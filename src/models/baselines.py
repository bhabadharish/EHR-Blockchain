"""Baseline Models Suite for Intrusion Detection (Phase 11).

Trains standard comparative baselines under identical evaluation protocol:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees
5. Multi-Layer Perceptron (MLP)
"""

import os
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from src.evaluation.metrics import compute_binary_metrics


class BaselineSuite:
    """Orchestrates baseline training and evaluation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Logistic_Regression": LogisticRegression(
                C=1.0, max_iter=500, random_state=random_state, n_jobs=-1
            ),
            "Decision_Tree": DecisionTreeClassifier(
                max_depth=12, random_state=random_state
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1
            ),
            "Extra_Trees": ExtraTreesClassifier(
                n_estimators=100, max_depth=12, random_state=random_state, n_jobs=-1
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=150, random_state=random_state, early_stopping=True
            ),
        }
        self.fitted_models: Dict[str, Any] = {}

    def train_and_evaluate(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, Any], float]:
        """Train a single baseline model and evaluate on test data."""
        clf = self.models[name]
        start_t = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - start_t

        self.fitted_models[name] = clf

        # Inference
        probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        preds = clf.predict(X_test)

        metrics = compute_binary_metrics(y_test, preds, probs)
        metrics["Model"] = name
        metrics["Train_Time_s"] = round(train_time, 3)

        return metrics, train_time

    def save_models(self, output_dir: str = "models/base/baselines") -> None:
        """Save fitted baseline checkpoints."""
        os.makedirs(output_dir, exist_ok=True)
        for name, clf in self.fitted_models.items():
            path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(clf, path)
