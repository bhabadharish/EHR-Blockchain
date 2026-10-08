"""Unit tests for HAB-IDS components and security layers."""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    CICIOT_ATTACK_FAMILIES,
    EDGE_IIOT_ATTACK_FAMILIES,
    COMMON_CROSS_DOMAIN_FEATURES,
    STRICT_LEAKAGE_EXCLUSIONS
)
from src.preprocessing.pipeline import LeakageFreePreprocessor
from src.ensemble.meta_learner import compute_meta_features
from src.security.pqc_layer import PQCSecurityLayer
from src.security.blockchain_adapter import BlockchainAuditAdapter
from src.security.fhir_risk_engine import FHIRRiskEngine


def test_schema_and_leakage_exclusions():
    assert "frame.time" in STRICT_LEAKAGE_EXCLUSIONS
    assert "timestamp" in STRICT_LEAKAGE_EXCLUSIONS
    assert "ip.src_host" in STRICT_LEAKAGE_EXCLUSIONS
    assert len(COMMON_CROSS_DOMAIN_FEATURES) == 7
    assert "Normal" in EDGE_IIOT_ATTACK_FAMILIES
    assert "BenignTraffic" in CICIOT_ATTACK_FAMILIES


def test_preprocessor_leakage_free():
    df_train = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "num2": [np.inf, 10.0, 20.0, 30.0],
        "cat1": ["a", "b", "a", "b"]
    })
    prep = LeakageFreePreprocessor(categorical_cols=["cat1"])
    X_tr = prep.fit_transform(df_train)
    assert not np.isnan(X_tr).any()
    assert not np.isinf(X_tr).any()

    # Test transform on unseen category
    df_test = pd.DataFrame({
        "num1": [5.0],
        "num2": [50.0],
        "cat1": ["unseen_category"]
    })
    X_te = prep.transform(df_test)
    assert X_te[0, 2] == -1.0  # unknown mapped to -1


def test_meta_features_computation():
    p_xgb = np.array([0.9, 0.1])
    p_lgb = np.array([0.8, 0.2])
    p_cat = np.array([0.85, 0.15])
    mf = compute_meta_features(p_xgb, p_lgb, p_cat)
    assert mf.shape == (2, 13)
    # Check d_range for first sample: max(0.9, 0.8, 0.85) - min = 0.1
    assert abs(mf[0, 7] - 0.1) < 1e-5


def test_pqc_layer():
    pqc = PQCSecurityLayer()
    pk, sk = pqc.ml_kem_keygen()
    assert len(pk) == pqc.ML_KEM_768_PK_SIZE
    ct, ss1 = pqc.ml_kem_encaps(pk)
    ss2 = pqc.ml_kem_decaps(ct, sk)
    assert ss1 == ss2

    pk_d, sk_d = pqc.ml_dsa_keygen()
    msg = b"EHR Observation Audit Record"
    sig = pqc.ml_dsa_sign(msg, sk_d)
    assert pqc.ml_dsa_verify(msg, sig, pk_d)


def test_blockchain_adapter(tmp_path):
    adapter = BlockchainAuditAdapter(vault_dir=str(tmp_path))
    res = adapter.record_security_audit(
        event_id="evt_001",
        fhir_resource_id="res_001",
        resource_hash="hash_001",
        risk_level="HIGH",
        model_version="1.0.0",
        prediction=1,
        digital_signature="sig_mock",
        encrypted_payload=b"test_encrypted_payload"
    )
    assert res["status"] == "COMMITTED"
    assert adapter.verify_audit_integrity(res["block_index"])


def test_fhir_risk_engine():
    engine = FHIRRiskEngine()
    eval_res = engine.evaluate_risk(
        ml_probability=0.92,
        auth_status=0,
        failed_auth_count=6,
        operation="delete",
        resource_type="DiagnosticReport",
        request_frequency=25.0,
        burst_score=0.85,
        historical_risk=0.75
    )
    assert eval_res["risk_category"] == "CRITICAL"
    assert "RC_ML_CONFIDENT_ATTACK" in eval_res["reason_codes"]
    assert "RC_BRUTE_FORCE_AUTHENTICATION" in eval_res["reason_codes"]
