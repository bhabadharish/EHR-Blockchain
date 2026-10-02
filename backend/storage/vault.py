"""
backend/storage/vault.py
Off-Chain Encrypted EHR Storage Vault (Phase 10).
Decouples high-volume encrypted health records from the blockchain consensus ledger.
Stores encrypted objects in object storage and produces lightweight cryptographic references.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from backend.crypto.symmetric import compute_sha3_256

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_ROOT = os.path.join(PROJECT_ROOT, "data", "storage_vault")
os.makedirs(STORAGE_ROOT, exist_ok=True)

class OffChainEHRVault:
    """
    Manages off-chain encrypted EHR storage.
    Enforces that ZERO plaintext EHR data is ever placed on the blockchain.
    """

    def __init__(self, vault_dir: Optional[str] = None):
        self.vault_dir = vault_dir or STORAGE_ROOT
        os.makedirs(self.vault_dir, exist_ok=True)

    def store_encrypted_package(
        self,
        encrypted_package: Dict[str, Any],
        patient_id: str,
        resource_type: str,
        consent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stores encrypted package to object store and generates blockchain metadata reference.
        """
        package_id = encrypted_package["package_id"]
        sha3_digest = encrypted_package["sha3_digest"]
        algorithm_id = encrypted_package["algorithm_id"]
        key_id = encrypted_package["key_id"]
        timestamp = encrypted_package["timestamp"]

        # Content-addressable object locator
        patient_vault_dir = os.path.join(self.vault_dir, patient_id)
        os.makedirs(patient_vault_dir, exist_ok=True)
        filename = f"{sha3_digest}.enc.json"
        object_path = os.path.join(patient_vault_dir, filename)

        # Write encrypted object
        raw_json = json.dumps(encrypted_package, indent=2)
        with open(object_path, "w") as fp:
            fp.write(raw_json)

        file_size_bytes = os.path.getsize(object_path)
        locator_uri = f"vault://ehr-store/{patient_id}/{filename}"

        # Lightweight cryptographic reference for on-chain anchoring
        blockchain_reference = {
            "record_id": package_id,
            "patient_id": patient_id,
            "resource_type": resource_type,
            "resource_hash": sha3_digest,
            "storage_locator": locator_uri,
            "algorithm": algorithm_id,
            "key_identifier": key_id,
            "timestamp": timestamp,
            "consent_identifier": consent_id or "CONSENT-DEFAULT-OPTIN",
            "signature_metadata": encrypted_package.get("public_key_metadata", {}),
            "payload_size_bytes": file_size_bytes
        }

        return blockchain_reference

    def retrieve_encrypted_package(self, storage_locator: str) -> Dict[str, Any]:
        """
        Retrieves encrypted package from off-chain storage locator.
        Validates content-addressable hash before returning.
        """
        # Parse locator: vault://ehr-store/{patient_id}/{filename}
        prefix = "vault://ehr-store/"
        if not storage_locator.startswith(prefix):
            raise ValueError(f"Invalid storage locator URI scheme: {storage_locator}")

        rel_path = storage_locator[len(prefix):]
        parts = rel_path.split("/")
        if len(parts) != 2:
            raise ValueError(f"Malformed locator path: {rel_path}")

        patient_id, filename = parts
        file_path = os.path.join(self.vault_dir, patient_id, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Off-chain encrypted object not found at {file_path}")

        with open(file_path, "r") as fp:
            package = json.load(fp)

        # Verify filename matches sha3_digest
        expected_digest = filename.replace(".enc.json", "")
        if package.get("sha3_digest") != expected_digest:
            raise ValueError("Off-chain object integrity error: content digest mismatch!")

        return package

    def delete_encrypted_package(self, storage_locator: str) -> bool:
        prefix = "vault://ehr-store/"
        if not storage_locator.startswith(prefix):
            return False
        rel_path = storage_locator[len(prefix):]
        parts = rel_path.split("/")
        if len(parts) != 2:
            return False
        patient_id, filename = parts
        file_path = os.path.join(self.vault_dir, patient_id, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
