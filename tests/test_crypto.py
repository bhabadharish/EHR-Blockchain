"""
tests/test_crypto.py
Unit tests for cryptographic suites: PQC (ML-KEM-768/1024, ML-DSA-65/87),
Classical (ECDH P-256, ECDSA), and Hybrid modes (Phases 7 & 8).
"""

import pytest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.crypto.engine import CryptoAgilityEngine
from backend.crypto.pqc import PQCStandardSuite, PQCHighSuite
from backend.crypto.classical import ClassicalSuite
from backend.crypto.hybrid import HybridSuite
from backend.crypto.symmetric import compute_sha3_256, encrypt_aes_256_gcm, decrypt_aes_256_gcm

@pytest.fixture
def sample_clinical_bundle():
    return {
        "resourceType": "Bundle",
        "id": "bundle-crypto-test",
        "type": "transaction",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pt-crypto-01",
                    "gender": "female",
                    "birthDate": "1985-07-20"
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-crypto-01",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                    "valueQuantity": {"value": 125.0, "unit": "mm[Hg]"}
                }
            }
        ]
    }

def test_pqc_standard_suite_primitives():
    suite = PQCStandardSuite()
    assert "MLKEM768" in suite.suite_id
    pk, sk = suite.generate_kem_keypair()
    assert len(pk) == 1184
    ct, ss1 = suite.encapsulate(pk)
    assert len(ct) == 1088
    ss2 = suite.decapsulate(sk, ct)
    assert ss1 == ss2
    assert len(ss1) == 32

    # Signatures
    spk, ssk = suite.generate_sign_keypair()
    msg = b"FHIR-PQC-INTEGRITY-CHECK"
    sig = suite.sign(ssk, msg)
    assert suite.verify(spk, msg, sig) is True
    assert suite.verify(spk, b"tampered message", sig) is False

def test_pqc_high_suite_primitives():
    suite = PQCHighSuite()
    assert "MLKEM1024" in suite.suite_id
    pk, sk = suite.generate_kem_keypair()
    assert len(pk) == 1568
    ct, ss1 = suite.encapsulate(pk)
    assert len(ct) == 1568
    ss2 = suite.decapsulate(sk, ct)
    assert ss1 == ss2

    spk, ssk = suite.generate_sign_keypair()
    msg = b"FHIR-HIGH-SECURITY-CHECK"
    sig = suite.sign(ssk, msg)
    assert suite.verify(spk, msg, sig) is True

def test_classical_suite_primitives():
    suite = ClassicalSuite()
    pk, sk = suite.generate_kem_keypair()
    ct, ss1 = suite.encapsulate(pk)
    ss2 = suite.decapsulate(sk, ct)
    assert ss1 == ss2
    assert len(ss1) == 32

    spk, ssk = suite.generate_sign_keypair()
    msg = b"CLASSICAL-MESSAGE"
    sig = suite.sign(ssk, msg)
    assert suite.verify(spk, msg, sig) is True
    assert suite.verify(spk, b"bad", sig) is False

def test_hybrid_suite_primitives():
    suite = HybridSuite()
    pk, sk = suite.generate_kem_keypair()
    ct, ss1 = suite.encapsulate(pk)
    ss2 = suite.decapsulate(sk, ct)
    assert ss1 == ss2
    assert len(ss1) == 32

    spk, ssk = suite.generate_sign_keypair()
    msg = b"HYBRID-AUTHENTICATED-MESSAGE"
    sig = suite.sign(ssk, msg)
    assert suite.verify(spk, msg, sig) is True

def test_aes_256_gcm_authenticated_encryption():
    key = b"0" * 32
    plaintext = b"Sensitive EHR clinical observations"
    aad = b"Associated-Metadata-123"

    ct, nonce, tag = encrypt_aes_256_gcm(plaintext, key, associated_data=aad)
    assert len(nonce) == 12
    assert len(tag) == 16

    decrypted = decrypt_aes_256_gcm(ct, nonce, tag, key, associated_data=aad)
    assert decrypted == plaintext

    # Tamper with ciphertext
    with pytest.raises(Exception):
        decrypt_aes_256_gcm(ct[:-1] + b"\x00", nonce, tag, key, associated_data=aad)

    # Tamper with AAD
    with pytest.raises(Exception):
        decrypt_aes_256_gcm(ct, nonce, tag, key, associated_data=b"Wrong-AAD")

@pytest.mark.parametrize("suite_name", ["PQC-MLKEM768", "PQC-MLKEM1024", "CLASSICAL", "HYBRID"])
def test_crypto_agility_engine_roundtrip(sample_clinical_bundle, suite_name):
    suite = CryptoAgilityEngine.get_suite(suite_name)
    recip_pk, recip_sk = suite.generate_kem_keypair()
    sender_pk, sender_sk = suite.generate_sign_keypair()

    package = CryptoAgilityEngine.package_fhir_resource(
        sample_clinical_bundle, recip_pk, sender_sk, suite_name=suite_name
    )

    assert package["algorithm_id"] == suite.suite_id
    assert "kem_ciphertext" in package
    assert "aes_ciphertext" in package
    assert "signature" in package
    assert "sha3_digest" in package

    # Unpackage and verify
    recovered_fhir, audit = CryptoAgilityEngine.unpackage_fhir_resource(
        package, recip_sk, sender_pk
    )

    assert audit["status"] == "VALID"
    assert audit["tampered"] is False
    assert audit["signature_valid"] is True
    assert audit["gcm_tag_valid"] is True
    assert audit["sha3_digest_valid"] is True
    assert recovered_fhir["id"] == sample_clinical_bundle["id"]
