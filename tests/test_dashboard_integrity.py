import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.fhir.resources import FHIRResourceManager
from src.crypto.pqc import PostQuantumCryptoEngine
from src.crypto.crypto_agility import CryptoAgilityEngine
from src.blockchain.client import FabricEHRClient
from src.security.response_engine import ThreatAwareResponseEngine
from src.security.fhir_risk_engine import FHIRRiskEngine

def test_result_registry_exists_and_valid():
    assert os.path.exists("models/registry.json")
    with open("models/registry.json") as f:
        reg = json.load(f)
    assert reg.get("model_name") == "HAB-IDS"
    assert "artifacts" in reg
    assert "xgboost" in reg["artifacts"]

def test_experiment_manifest_exists():
    assert os.path.exists("results/experiment_manifest.json")
    with open("results/experiment_manifest.json") as f:
        man = json.load(f)
    assert "git_commit" in man
    assert "hardware" in man
    assert "dataset_hashes" in man

def test_threshold_specification():
    assert os.path.exists("models/final/decision_threshold.json")
    with open("models/final/decision_threshold.json") as f:
        tm = json.load(f)
    opt_t = tm.get("optimal_threshold", 0.0)
    assert 0.0 < opt_t < 1.0

def test_dashboard_artifacts_loadable():
    # Verify core JSON benchmark artifacts load properly
    with open("results/benchmark/blockchain_benchmark.json") as f:
        bc_j = json.load(f)
    assert len(bc_j) == 10
    
    with open("results/benchmark/crypto_benchmark.json") as f:
        cr_j = json.load(f)
    assert len(cr_j) >= 10

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

def test_blockchain_tamper_detection(tmp_path):
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
