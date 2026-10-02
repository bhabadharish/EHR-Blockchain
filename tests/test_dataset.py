"""
tests/test_dataset.py
Tests for dataset separation, non-leaking preprocessing, and telemetry loading (Phases 14 & 16).
"""

import pytest
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.security.telemetry import TelemetryDatasetManager

def test_telemetry_pipeline_non_leaking():
    data = TelemetryDatasetManager.load_and_preprocess(seed=42)
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train_multi"]
    y_test = data["y_test_multi"]

    # Verify non-empty
    assert len(X_train) > 50000
    assert len(X_val) > 10000
    assert len(X_test) > 10000

    # Verify no NaN or Inf
    assert np.isnan(X_train).sum() == 0
    assert np.isnan(X_val).sum() == 0
    assert np.isnan(X_test).sum() == 0
    assert np.isinf(X_train).sum() == 0
    assert np.isinf(X_test).sum() == 0

    # Verify class count
    assert len(data["class_names"]) == 15
    assert len(data["feature_names"]) == 50

    # Verify ratio bounds: ~70/15/15
    total = len(X_train) + len(X_val) + len(X_test)
    assert 0.68 <= len(X_train) / total <= 0.72
    assert 0.13 <= len(X_val) / total <= 0.17
    assert 0.13 <= len(X_test) / total <= 0.17

def test_dataset_manifest_integrity():
    import json
    import hashlib
    manifest_path = os.path.join(PROJECT_ROOT, "data", "metadata", "dataset_manifest.json")
    assert os.path.exists(manifest_path), "dataset_manifest.json must exist"
    
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
    
    datasets = manifest.get("datasets", {})
    assert "Edge-IIoTset" in datasets
    assert "Synthea-FHIR-R4" in datasets
    assert "MIMIC-IV-on-FHIR" in datasets

    # Verify Edge-IIoTset checksum
    edge_meta = datasets["Edge-IIoTset"]
    assert os.path.exists(edge_meta["local_path"])
    sha = hashlib.sha256()
    with open(edge_meta["local_path"], "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha.update(chunk)
    assert sha.hexdigest() == edge_meta["checksum_sha256"]

def test_strict_dataset_separation():
    import pandas as pd
    csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "ML-EdgeIIoT-dataset.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path, nrows=50)
    
    # Confirm zero PHI or clinical terminology in cybersecurity dataset
    phi_terms = {"patient", "ssn", "mrn", "birthdate", "blood_pressure", "cholesterol", "doctor", "diagnosis"}
    cols_lower = [c.lower() for c in df.columns]
    leaked = [t for t in phi_terms if any(t in c for c in cols_lower)]
    assert len(leaked) == 0, f"Leaked clinical identifiers in cybersecurity data: {leaked}"

def test_mimic_governance_provenance():
    import json
    manifest_path = os.path.join(PROJECT_ROOT, "data", "metadata", "dataset_manifest.json")
    with open(manifest_path, "r") as fp:
        manifest = json.load(fp)
    mimic_meta = manifest["datasets"]["MIMIC-IV-on-FHIR"]
    assert "CITI" in mimic_meta["access_requirement"]
    assert mimic_meta["status"] in ["REQUIRES_CREDENTIALED_APPROVAL", "LOCAL_CREDENTIALED_COPY_FOUND"]

