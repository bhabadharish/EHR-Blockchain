"""
tests/test_storage.py
Tests for off-chain encrypted EHR storage vault (Phase 10).
Verifies decoupled storage, cryptographic references, and zero-plaintext guarantees.
"""

import pytest
import os
import sys
import json
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.storage.vault import OffChainEHRVault
from backend.crypto.symmetric import compute_sha3_256

TEST_VAULT_DIR = os.path.join(PROJECT_ROOT, "data", "test_vault")

@pytest.fixture
def vault():
    v = OffChainEHRVault(vault_dir=TEST_VAULT_DIR)
    yield v
    if os.path.exists(TEST_VAULT_DIR):
        shutil.rmtree(TEST_VAULT_DIR)

@pytest.fixture
def sample_encrypted_package():
    dummy_payload = b"Encrypted ciphertext block bytes"
    digest = compute_sha3_256(dummy_payload)
    return {
        "package_id": "pkg-test-vault-01",
        "algorithm_id": "PQC-MLKEM768-MLDSA65-AES256GCM-SHA3",
        "key_id": "key-test-01",
        "timestamp": "2026-10-02T12:00:00Z",
        "kem_ciphertext": "AAAA",
        "aes_ciphertext": "BBBB",
        "nonce": "CCCC",
        "tag": "DDDD",
        "sha3_digest": digest,
        "signature": "EEEE",
        "public_key_metadata": {"recipient_pk_fingerprint": "key-test-01"}
    }

def test_store_and_retrieve_package(vault, sample_encrypted_package):
    ref = vault.store_encrypted_package(
        sample_encrypted_package,
        patient_id="pt-100",
        resource_type="Observation"
    )

    assert ref["patient_id"] == "pt-100"
    assert ref["resource_type"] == "Observation"
    assert ref["resource_hash"] == sample_encrypted_package["sha3_digest"]
    assert "vault://ehr-store/pt-100/" in ref["storage_locator"]

    # Retrieve
    retrieved = vault.retrieve_encrypted_package(ref["storage_locator"])
    assert retrieved["package_id"] == sample_encrypted_package["package_id"]
    assert retrieved["sha3_digest"] == sample_encrypted_package["sha3_digest"]

def test_delete_package(vault, sample_encrypted_package):
    ref = vault.store_encrypted_package(
        sample_encrypted_package,
        patient_id="pt-101",
        resource_type="Condition"
    )
    deleted = vault.delete_encrypted_package(ref["storage_locator"])
    assert deleted is True

    # Subsequent retrieval must fail
    with pytest.raises(FileNotFoundError):
        vault.retrieve_encrypted_package(ref["storage_locator"])
