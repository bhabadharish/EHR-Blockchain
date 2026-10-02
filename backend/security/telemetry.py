"""
backend/security/telemetry.py
Security Telemetry Abstraction Layer and Non-Leaking Data Pipeline (Phases 14 & 16).
Ingests Edge-IIoTset, removes leakage identifiers, performs strict stratified train/val/test splits,
fits transformers strictly on training data, and exposes normalized feature vectors.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Columns that cause data leakage / memorization / overfitting
LEAKAGE_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "tcp.payload",
    "tcp.options",
    "http.request.full_uri",
    "http.file_data",
    "http.request.uri.query",
    "tcp.srcport"
]

class TelemetryDatasetManager:
    """
    Manages non-leaking preprocessing and reproducible train/val/test splits for Edge-IIoTset.
    """

    @classmethod
    def load_and_preprocess(
        cls,
        seed: int = 42,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        split_meta_path = os.path.join(SPLITS_DIR, f"split_metadata_seed_{seed}.json")
        split_npz_path = os.path.join(SPLITS_DIR, f"telemetry_splits_seed_{seed}.npz")

        if not force_recompute and os.path.exists(split_meta_path) and os.path.exists(split_npz_path):
            print(f"[CACHE] Loading pre-split telemetry data from {split_npz_path}...")
            npz = np.load(split_npz_path)
            with open(split_meta_path) as fp:
                meta = json.load(fp)
            return {
                "X_train": npz["X_train"],
                "y_train_multi": npz["y_train_multi"],
                "y_train_bin": npz["y_train_bin"],
                "X_val": npz["X_val"],
                "y_val_multi": npz["y_val_multi"],
                "y_val_bin": npz["y_val_bin"],
                "X_test": npz["X_test"],
                "y_test_multi": npz["y_test_multi"],
                "y_test_bin": npz["y_test_bin"],
                "feature_names": meta["feature_names"],
                "class_names": meta["class_names"],
                "metadata": meta
            }

        raw_csv = os.path.join(RAW_DIR, "ML-EdgeIIoT-dataset.csv")
        if not os.path.exists(raw_csv):
            raise FileNotFoundError(f"Raw telemetry CSV not found at {raw_csv}")

        print(f"[INGESTION] Loading {raw_csv}...")
        df = pd.read_csv(raw_csv, low_memory=False)

        # 1. Drop leakage columns
        df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

        # 2. Duplicate removal
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dedup_count = len(df)
        print(f"  Removed {initial_count - dedup_count:,} duplicate records. Retained {dedup_count:,} clean samples.")

        # 3. Separate features and target labels
        y_multi_raw = df["Attack_type"]
        y_bin_raw = df["Attack_label"].values.astype(np.int64)
        X_df = df.drop(columns=["Attack_label", "Attack_type"])

        # Label encode multiclass targets
        label_encoder = LabelEncoder()
        y_multi_encoded = label_encoder.fit_transform(y_multi_raw)
        class_names = list(label_encoder.classes_)
        num_classes = len(class_names)

        # 4. Split data FIRST: 70% Train, 15% Val, 15% Test
        temp_ratio = val_ratio + test_ratio
        X_train_df, X_temp_df, y_train_m, y_temp_m, y_train_b, y_temp_b = train_test_split(
            X_df, y_multi_encoded, y_bin_raw,
            test_size=temp_ratio,
            random_state=seed,
            stratify=y_multi_encoded
        )

        val_fraction = val_ratio / temp_ratio
        X_val_df, X_test_df, y_val_m, y_test_m, y_val_b, y_test_b = train_test_split(
            X_temp_df, y_temp_m, y_temp_b,
            test_size=(1.0 - val_fraction),
            random_state=seed,
            stratify=y_temp_m
        )

        # 5. Fit scalers and encoders STRICTLY on training split
        cat_cols = [c for c in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[c])]
        num_cols = [c for c in X_df.columns if c not in cat_cols]

        feature_names = num_cols + cat_cols

        # Categorical encoder (fit on train only)
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit(X_train_df[cat_cols].astype(str))

        X_train_cat = cat_encoder.transform(X_train_df[cat_cols].astype(str))
        X_val_cat = cat_encoder.transform(X_val_df[cat_cols].astype(str))
        X_test_cat = cat_encoder.transform(X_test_df[cat_cols].astype(str))

        # Numerical scaler (fit on train only)
        num_scaler = StandardScaler()
        num_scaler.fit(X_train_df[num_cols].astype(np.float32))

        X_train_num = num_scaler.transform(X_train_df[num_cols].astype(np.float32))
        X_val_num = num_scaler.transform(X_val_df[num_cols].astype(np.float32))
        X_test_num = num_scaler.transform(X_test_df[num_cols].astype(np.float32))

        # Combine processed numerical and encoded categoricals
        X_train = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
        X_val = np.hstack([X_val_num, X_val_cat]).astype(np.float32)
        X_test = np.hstack([X_test_num, X_test_cat]).astype(np.float32)

        # Save fitted preprocessors
        joblib.dump(num_scaler, os.path.join(MODELS_DIR, f"num_scaler_seed_{seed}.joblib"))
        joblib.dump(cat_encoder, os.path.join(MODELS_DIR, f"cat_encoder_seed_{seed}.joblib"))
        joblib.dump(label_encoder, os.path.join(MODELS_DIR, f"label_encoder_seed_{seed}.joblib"))

        # Save arrays to compressed npz
        np.savez_compressed(
            split_npz_path,
            X_train=X_train, y_train_multi=y_train_m, y_train_bin=y_train_b,
            X_val=X_val, y_val_multi=y_val_m, y_val_bin=y_val_b,
            X_test=X_test, y_test_multi=y_test_m, y_test_bin=y_test_b
        )

        metadata = {
            "dataset_name": "Edge-IIoTset",
            "seed": seed,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "num_features": int(X_train.shape[1]),
            "num_classes": int(num_classes),
            "class_names": class_names,
            "feature_names": feature_names,
            "non_leaking_guarantee": "Scalers and encoders fit strictly on train split."
        }

        with open(split_meta_path, "w") as fp:
            json.dump(metadata, fp, indent=2)

        print(f"[PREPROCESSED] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return {
            "X_train": X_train,
            "y_train_multi": y_train_m,
            "y_train_bin": y_train_b,
            "X_val": X_val,
            "y_val_multi": y_val_m,
            "y_val_bin": y_val_b,
            "X_test": X_test,
            "y_test_multi": y_test_m,
            "y_test_bin": y_test_b,
            "feature_names": feature_names,
            "class_names": class_names,
            "metadata": metadata
        }
