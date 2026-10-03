import os
import sys
import json
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.validation.result_consistency import ResultConsistencyEngine
from dashboard.inference.model_loader import FrozenModelLoader
from src.fhir.resources import FHIRResourceManager
from src.crypto.pqc import PostQuantumCryptoEngine
from src.crypto.crypto_agility import CryptoAgilityEngine
from src.blockchain.client import FabricEHRClient
from src.security.response_engine import ThreatAwareResponseEngine

def test_result_registry_exists_and_valid():
    assert os.path.exists("results/experiment_registry.json")
    with open("results/experiment_registry.json") as f:
        reg = json.load(f)
    assert "models" in reg
    assert "CA-HTDNet" in reg["models"]
    assert reg["experiment_id"] in ["CAHTDNET_FINAL_V001", "CAHTDNet_final_locked_v001", "CAHTDNET_V2_001"]

def test_model_and_preprocessor_hashes():
    loader = FrozenModelLoader()
    assert loader.integrity_status["model_status"] == "VERIFIED"
    assert loader.integrity_status["preprocessor_status"] == "VERIFIED"
    assert 0.0 < loader.optimal_threshold < 1.0

def test_metric_reconciliation_zero_discrepancy():
    engine = ResultConsistencyEngine()
    report = engine.verify_all_models()
    assert report["overall_status"] == "PASS"
    assert len(report["discrepancies"]) == 0

def test_class_mapping_consistency():
    assert os.path.exists("data/metadata/label_mapping.json")
    with open("data/metadata/label_mapping.json") as f:
        lm = json.load(f)
    assert "binary_classes" in lm
    assert lm["binary_classes"] == ["Benign", "Attack"]
    assert "detailed_classes" in lm
    assert len(lm["detailed_classes"]) == 34

def test_threshold_specification():
    assert os.path.exists("models/proposed/threshold.json")
    with open("models/proposed/threshold.json") as f:
        tm = json.load(f)
    assert 0.0 < tm["optimal_threshold"] < 1.0
    assert tm.get("test_set_used_in_selection", tm.get("test_set_used")) is False

def test_frozen_inference_no_training():
    loader = FrozenModelLoader()
    sample = {
        "flow_duration": 0.05, "packet_count": 10, "byte_count": 1200,
        "packet_rate": 200.0, "byte_rate": 24000.0, "dst_port": 443, "protocol": 6,
        "resource_sensitivity": 0.5, "auth_status": 1, "failed_auth_count": 0,
        "request_frequency": 2.0, "burst_score": 0.1, "historical_risk": 0.05,
        "user_role": "doctor", "resource_type": "Observation", "operation": "read"
    }
    res = loader.predict(sample)
    assert res["predicted_class"] in ["Benign / Normal", "Cyber Threat"]
    assert 0.0 <= res["threat_probability"] <= 1.0
    assert res["decision"] in ["ALLOW", "REVIEW", "BLOCK"]

def test_fhir_generation_and_validation():
    pat = FHIRResourceManager.create_patient("p-100", "Alice Smith", "female", "1980-05-12")
    assert pat["resourceType"] == "Patient"
    assert pat["id"] == "p-100"

    obs = FHIRResourceManager.create_observation("obs-1", "p-100", "8867-4", "Heart rate", 75.0, "beats/min")
    assert obs["resourceType"] == "Observation"
    assert obs["valueQuantity"]["value"] == 75.0

def test_crypto_layer_ml_kem_and_dsa():
    pqc = PostQuantumCryptoEngine()
    pk, sk = pqc.ml_kem_generate_keypair()
    ct, ss1 = pqc.ml_kem_encapsulate(pk)
    ss2 = pqc.ml_kem_decapsulate(ct, sk)
    assert ss1 == ss2
    assert len(ss1) == 32

    msg = b"Integrity Proof"
    dpk, dsk = pqc.ml_dsa_generate_keypair()
    sig = pqc.ml_dsa_sign(msg, dsk)
    assert pqc.ml_dsa_verify(msg, sig, dpk) is True

def test_blockchain_tamper_detection():
    client = FabricEHRClient()
    res_obj = {"resourceType": "Observation", "id": "test-obs-pytest", "value": 98.6}
    reg = client.register_ehr_exchange("dr_test", "pt-test", "test-obs-pytest", res_obj, 0.05)
    
    # Clean check
    clean_v = client.verify_resource_integrity("test-obs-pytest", reg["vault_pointer"])
    assert clean_v["verified"] is True

    # Mutate disk
    with open(reg["vault_pointer"], "r+b") as f:
        f.seek(16)
        f.write(b"\xff\xff\xff\xff")
    
    tampered_v = client.verify_resource_integrity("test-obs-pytest", reg["vault_pointer"])
    assert tampered_v["verified"] is False
