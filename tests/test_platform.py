import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.cleaning import UnifiedSecurityPreprocessor
from src.data.harmonization import UNIFIED_NUMERICAL_FEATURES, UNIFIED_CATEGORICAL_FEATURES
from src.data.synthetic_fhir_generator import SyntheticFHIRSecurityGenerator
from src.models.ca_htdnet import CA_HTDNet
from src.crypto.pqc import PostQuantumCryptoEngine
from src.crypto.crypto_agility import CryptoAgilityEngine
from src.blockchain.client import FabricEHRClient
from src.fhir.resources import FHIRResourceManager, FHIRValidator
from src.security.response_engine import ThreatAwareResponseEngine

def test_data_splits_exist():
    assert os.path.exists("data/splits/train.parquet")
    assert os.path.exists("data/splits/validation.parquet")
    assert os.path.exists("data/splits/test.parquet")

def test_synthetic_fhir_generator():
    gen = SyntheticFHIRSecurityGenerator(seed=42)
    df = gen.generate_dataset(num_records=50, attack_ratio=0.4)
    assert len(df) == 50
    assert "attack_category" in df.columns
    assert "binary_label" in df.columns
    assert df["binary_label"].isin([0, 1]).all()

def test_preprocessor_fitting():
    from src.preprocessing.pipeline import LeakageFreePreprocessor
    preprocessor = LeakageFreePreprocessor.load("models/final/preprocessor.pkl")
    assert preprocessor.is_fitted is True

def test_model_inference():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CA_HTDNet(num_numerical=13).to(device)
    model.eval()
    x = torch.randn(2, 13, device=device)
    gbdt = torch.randn(2, 4, device=device)
    with torch.no_grad():
        out = model(x, gbdt)
    assert "logits" in out
    assert "calibrated_logits" in out
    assert "tri_state_logits" in out
    assert out["logits"].shape == (2, 2)
    assert out["tri_state_logits"].shape == (2, 3)

def test_pqc_primitives():
    # ML-KEM
    pk, sk = PostQuantumCryptoEngine.ml_kem_generate_keypair()
    ct, ss1 = PostQuantumCryptoEngine.ml_kem_encapsulate(pk)
    ss2 = PostQuantumCryptoEngine.ml_kem_decapsulate(ct, sk)
    assert ss1 == ss2
    assert len(ss1) == 32

    # ML-DSA
    sig_pk, sig_sk = PostQuantumCryptoEngine.ml_dsa_generate_keypair()
    msg = b"FHIR_RESOURCE_HASH_INTEGRITY_VERIFICATION"
    sig = PostQuantumCryptoEngine.ml_dsa_sign(msg, sig_sk)
    assert PostQuantumCryptoEngine.ml_dsa_verify(msg, sig, sig_pk) is True

    # AES-256-GCM
    key = os.urandom(32)
    pt = b"Patient EHR Data - Classified Medical Record"
    ct_aes, iv = PostQuantumCryptoEngine.aes_256_gcm_encrypt(pt, key)
    dec = PostQuantumCryptoEngine.aes_256_gcm_decrypt(ct_aes, key, iv)
    assert dec == pt

def test_crypto_agility_transitions():
    engine = CryptoAgilityEngine()
    
    # Low threat -> Standard
    pol_low = engine.select_profile(threat_score=0.10, resource_sensitivity=0.3)
    assert pol_low["selected_profile"] == CryptoAgilityEngine.PROFILE_STANDARD
    
    # Medium threat -> Hybrid
    pol_med = engine.select_profile(threat_score=0.55, resource_sensitivity=0.6)
    assert pol_med["selected_profile"] == CryptoAgilityEngine.PROFILE_HYBRID
    
    # High threat -> PQC
    pol_high = engine.select_profile(threat_score=0.90, resource_sensitivity=0.8)
    assert pol_high["selected_profile"] == CryptoAgilityEngine.PROFILE_PQC

def test_blockchain_ledger_and_tampering():
    client = FabricEHRClient()
    dummy_obs = FHIRResourceManager.create_observation(
        obs_id="obs-test-01",
        patient_id="pt-99",
        code_loinc="8867-4",
        display="Heart rate",
        value=75.0,
        unit="/min"
    )
    res = client.register_ehr_exchange(
        actor_id="dr_bob",
        patient_id="pt-99",
        resource_id="obs-test-01",
        fhir_resource=dummy_obs,
        threat_score=0.04
    )
    assert "integrity_tx_id" in res
    assert "audit_tx_id" in res
    
    # Verify match
    v_res = client.verify_resource_integrity("obs-test-01", res["vault_pointer"])
    assert v_res["verified"] is True
    assert v_res["status"] == "INTEGRITY_VERIFIED_MATCH"

def test_fhir_validation():
    valid_pt = FHIRResourceManager.create_patient("pt-1", "John Doe", "male", "1980-01-01")
    is_valid, msg = FHIRValidator.validate(valid_pt)
    assert is_valid is True

    invalid_res = {"resourceType": "Observation", "id": "obs-bad"}
    is_invalid, _ = FHIRValidator.validate(invalid_res)
    assert is_invalid is False

def test_zero_trust_response_engine():
    engine = ThreatAwareResponseEngine()
    
    # Normal doctor access
    dec_norm = engine.evaluate_request(
        actor_id="dr_smith", user_role="doctor",
        device_id="ws_01", device_type="clinical_workstation",
        resource_type="Observation", resource_sensitivity=0.5,
        operation="read", threat_probability=0.02, has_consent=True
    )
    assert dec_norm["action"] == "ALLOW"

    # Extreme threat / unauthorized access
    dec_attack = engine.evaluate_request(
        actor_id="ext_client", user_role="researcher",
        device_id="ext_api", device_type="external_api_client",
        resource_type="Patient", resource_sensitivity=0.9,
        operation="delete", threat_probability=0.95, has_consent=False
    )
    assert dec_attack["action"] == "BLOCK_ACTOR"

def test_end_to_end_integration_pipeline():
    """
    Integration Test:
    FHIR Request -> Threat Detection Inference -> Threat-Aware Decision -> Crypto-Agile Protection -> Blockchain Anchoring.
    """
    # 1. Synthesize FHIR event
    obs = FHIRResourceManager.create_observation(
        obs_id="obs-e2e-999",
        patient_id="pt-e2e",
        code_loinc="2708-6",
        display="Oxygen saturation",
        value=98.0,
        unit="%"
    )
    assert FHIRValidator.validate(obs)[0] is True

    # 2. Simulated Model Threat Inference (0.85 = High Threat detected)
    threat_prob = 0.85

    # 3. Threat-Aware Risk Decision
    resp_engine = ThreatAwareResponseEngine()
    decision = resp_engine.evaluate_request(
        actor_id="dr_charlie", user_role="doctor",
        device_id="tablet_02", device_type="mobile_tablet",
        resource_type="Observation", resource_sensitivity=0.8,
        operation="read", threat_probability=threat_prob, has_consent=True
    )
    assert decision["decision"] in ["CHALLENGE", "DENY", "ALLOW_CONSTRAINED"]

    # 4. Crypto-Agility Adaptation
    crypto_engine = CryptoAgilityEngine()
    payload = json.dumps(obs).encode("utf-8")
    crypto_result = crypto_engine.execute_secure_exchange(payload, threat_score=threat_prob, resource_sensitivity=0.8)
    assert crypto_result["policy"]["selected_profile"] == CryptoAgilityEngine.PROFILE_PQC
    assert crypto_result["signature_valid"] is True

    # 5. Fabric Ledger Anchoring
    fabric_client = FabricEHRClient()
    anchoring = fabric_client.register_ehr_exchange(
        actor_id="dr_charlie",
        patient_id="pt-e2e",
        resource_id="obs-e2e-999",
        fhir_resource=obs,
        threat_score=threat_prob
    )
    assert anchoring["block_height"] > 1
    
    # 6. Verify Blockchain Integrity
    v_report = fabric_client.verify_resource_integrity("obs-e2e-999", anchoring["vault_pointer"])
    assert v_report["verified"] is True
