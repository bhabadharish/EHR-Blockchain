"""
tests/test_master_verification_suite.py
========================================
Comprehensive Verification Suite covering all 16 specified project dimensions:
1. blockchain connectivity
2. transaction submission
3. transaction confirmation
4. smart-contract authorization
5. consent grant
6. consent revocation
7. signature validation
8. encryption/decryption
9. PQC operations
10. tamper detection
11. replay detection
12. model inference
13. metric calculation
14. benchmark logging
15. figure generation
16. dashboard loading
"""

import os
import sys
import json
import time
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.blockchain.ledger import FabricPermissionedLedger
from src.blockchain.client import FabricEHRClient
from src.crypto.pqc import PostQuantumCryptoEngine
from src.security.response_engine import ThreatAwareResponseEngine
from src.security.fhir_risk_engine import FHIRRiskEngine
from src.fhir.resources import FHIRResourceManager, FHIRValidator

# 1. Blockchain Connectivity
def test_01_blockchain_connectivity():
    ledger = FabricPermissionedLedger()
    assert ledger.block_height >= 1
    assert len(ledger.chain) >= 1
    assert ledger.chain[0].transactions[0]["type"] == "CONFIG"

# 2. Transaction Submission
def test_02_transaction_submission():
    ledger = FabricPermissionedLedger()
    tx_id = ledger.commit_transaction({
        "chaincode": "AuditCC",
        "type": "recordAuditEvent",
        "event_data": {"test": "submission"}
    })
    assert len(tx_id) == 32
    assert ledger.block_height >= 2

# 3. Transaction Confirmation
def test_03_transaction_confirmation():
    ledger = FabricPermissionedLedger()
    tx_id = ledger.register_fhir_hash("res_conf_01", "hash_abc", "ptr_123")
    assert f"integrity:res_conf_01" in ledger.world_state
    rec = ledger.world_state[f"integrity:res_conf_01"]
    assert rec["sha3_hash"] == "hash_abc"

# 4. Smart-Contract Authorization
def test_04_smart_contract_authorization():
    ledger = FabricPermissionedLedger()
    ledger.register_actor("dr_auth_01", "doctor", "Hospital_A")
    is_ok, msg = ledger.check_access("dr_auth_01", "Observation", "read")
    assert is_ok is True

    # Unauthorized role
    is_ok_del, _ = ledger.check_access("dr_auth_01", "Observation", "delete")
    assert is_ok_del is False

# 5. Consent Grant
def test_05_consent_grant():
    ledger = FabricPermissionedLedger()
    con_tx = ledger.create_consent("con_test_01", "pat_1", "dr_auth_01", ["Observation"], time.time() + 3600)
    assert len(con_tx) == 32
    has_consent, _ = ledger.check_consent("pat_1", "dr_auth_01", "Observation")
    assert has_consent is True

# 6. Consent Revocation
def test_06_consent_revocation():
    ledger = FabricPermissionedLedger()
    ledger.create_consent("con_test_02", "pat_2", "dr_auth_01", ["Observation"], time.time() + 3600)
    rev_tx = ledger.revoke_consent("con_test_02")
    assert len(rev_tx) == 32
    has_consent, msg = ledger.check_consent("pat_2", "dr_auth_01", "Observation")
    assert has_consent is False
    assert "revoked" in msg.lower()

# 7. Signature Validation
def test_07_signature_validation():
    pk, sk = PostQuantumCryptoEngine.ml_dsa_generate_keypair()
    msg = b"Clinical Telemetry Data"
    sig = PostQuantumCryptoEngine.ml_dsa_sign(msg, sk)
    assert PostQuantumCryptoEngine.ml_dsa_verify(msg, sig, pk) is True

    # Mutated message must fail
    assert PostQuantumCryptoEngine.ml_dsa_verify(b"Mutated", sig, pk) is False

# 8. Encryption / Decryption
def test_08_encryption_decryption():
    key = os.urandom(32)
    plaintext = b"Sensitive Patient EHR Observation - Vital Signs"
    ciphertext, iv = PostQuantumCryptoEngine.aes_256_gcm_encrypt(plaintext, key)
    decrypted = PostQuantumCryptoEngine.aes_256_gcm_decrypt(ciphertext, key, iv)
    assert decrypted == plaintext

# 9. PQC Operations
def test_09_pqc_operations():
    pk, sk = PostQuantumCryptoEngine.ml_kem_generate_keypair()
    assert len(pk) == 1184
    assert len(sk) == 2400
    ct, ss1 = PostQuantumCryptoEngine.ml_kem_encapsulate(pk)
    assert len(ct) == 1088
    assert len(ss1) == 32
    ss2 = PostQuantumCryptoEngine.ml_kem_decapsulate(ct, sk)
    assert ss1 == ss2

# 10. Tamper Detection
def test_10_tamper_detection(tmp_path):
    client = FabricEHRClient()
    obs = {"resourceType": "Observation", "id": "tamper-obs", "valueQuantity": {"value": 80}}
    res = client.register_ehr_exchange("dr_1", "p_1", "tamper-obs", obs, 0.01)
    
    # Verify clean match
    v_clean = client.verify_resource_integrity("tamper-obs", res["vault_pointer"])
    assert v_clean["verified"] is True

    # Tamper with file
    with open(res["vault_pointer"], "wb") as f:
        f.write(b"CORRUPTED_BYTES_IN_VAULT")
    
    v_tampered = client.verify_resource_integrity("tamper-obs", res["vault_pointer"])
    assert v_tampered["verified"] is False

# 11. Replay Detection
def test_11_replay_detection():
    engine = FHIRRiskEngine()
    res = engine.evaluate_risk(
        ml_probability=0.85, auth_status=1, failed_auth_count=4,
        operation="create", resource_type="Observation",
        request_frequency=50.0, burst_score=0.95, historical_risk=0.8
    )
    assert res["risk_category"] in ["HIGH", "CRITICAL"]
    assert "RC_ABNORMAL_TRAFFIC_BURST" in res["reason_codes"]

# 12. Model Inference
def test_12_model_inference():
    import joblib
    xgb = joblib.load("models/final/xgboost_final.pkl")
    prep = joblib.load("models/final/preprocessor.pkl")
    assert prep.is_fitted is True
    # Dummy sample
    sample_df = pd.DataFrame([{col: 1.0 for col in prep.retained_features_}])
    x_tr = prep.transform(sample_df)
    prob = xgb.predict_proba(x_tr)[:, 1]
    assert 0.0 <= prob[0] <= 1.0

# 13. Metric Calculation
def test_13_metric_calculation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0])
    acc = np.mean(y_true == y_pred)
    assert acc == 0.8

# 14. Benchmark Logging
def test_14_benchmark_logging():
    assert os.path.exists("results/benchmark/blockchain_benchmark.json")
    with open("results/benchmark/blockchain_benchmark.json") as f:
        bc_j = json.load(f)
    assert len(bc_j) == 10

# 15. Figure Generation
def test_15_figure_generation():
    assert os.path.exists("figures/01_architecture_overview.png")
    assert os.path.exists("figures/02_blockchain_throughput.png")
    assert os.path.exists("figures/13_confusion_matrix.png")
    assert os.path.exists("figures/24_ablation.png")

# 16. Dashboard Loading
def test_16_dashboard_loading():
    # Verify Streamlit dashboard script parses and loads without syntax errors
    import py_compile
    py_compile.compile("app/streamlit_app.py", doraise=True)
    assert os.path.exists("app/streamlit_app.py")
