import os
import json
import time
from typing import Dict, Any, Tuple, Optional
from src.blockchain.ledger import FabricPermissionedLedger
from src.crypto.pqc import PostQuantumCryptoEngine

class OffChainEHRVault:
    """
    Off-Chain Encrypted EHR Storage Vault.
    Guarantees patient data privacy by storing strictly ciphertext off-chain,
    while anchoring cryptographic SHA3-256 integrity proofs into the Hyperledger Fabric ledger.
    """
    def __init__(self, vault_dir: str = "data/processed/offchain_vault"):
        self.vault_dir = vault_dir
        os.makedirs(vault_dir, exist_ok=True)
        self.symmetric_key = os.urandom(32)

    def store_resource(self, resource_id: str, fhir_json: Dict[str, Any]) -> Tuple[str, str, str]:
        canonical_bytes = json.dumps(fhir_json, sort_keys=True).encode("utf-8")
        integrity_hash = PostQuantumCryptoEngine.sha3_256_hash(canonical_bytes)
        
        # AES-256-GCM authenticated encryption
        ciphertext, iv = PostQuantumCryptoEngine.aes_256_gcm_encrypt(canonical_bytes, self.symmetric_key)
        
        # Save to off-chain store
        pointer = os.path.join(self.vault_dir, f"{resource_id}.enc")
        with open(pointer, "wb") as f:
            f.write(iv + ciphertext)
            
        return pointer, integrity_hash, canonical_bytes.decode("utf-8")

    def retrieve_resource(self, pointer: str) -> bytes:
        if not os.path.exists(pointer):
            raise FileNotFoundError(f"Vault pointer {pointer} does not exist.")
        with open(pointer, "rb") as f:
            raw = f.read()
        iv = raw[:12]
        ct = raw[12:]
        return PostQuantumCryptoEngine.aes_256_gcm_decrypt(ct, self.symmetric_key, iv)

class FabricEHRClient:
    """
    High-level Hyperledger Fabric Client for Healthcare EHR Exchange.
    Integrates ledger chaincodes, off-chain encrypted vault, and crypto-agility.
    """
    def __init__(self):
        self.ledger = FabricPermissionedLedger()
        self.vault = OffChainEHRVault()

    def register_ehr_exchange(
        self,
        actor_id: str,
        patient_id: str,
        resource_id: str,
        fhir_resource: Dict[str, Any],
        threat_score: float = 0.05
    ) -> Dict[str, Any]:
        """
        Executes end-to-end off-chain encryption and on-chain anchoring transaction.
        """
        pointer, sha3_hash, canonical_str = self.vault.store_resource(resource_id, fhir_resource)
        
        # On-chain integrity registration
        tx_integrity = self.ledger.register_fhir_hash(resource_id, sha3_hash, pointer)
        
        # On-chain audit trail
        tx_audit = self.ledger.record_audit_event(
            actor_id=actor_id,
            action=fhir_resource.get("resourceType", "Resource") + "_WRITE",
            resource_id=resource_id,
            outcome="SUCCESS",
            threat_score=threat_score
        )

        return {
            "resource_id": resource_id,
            "sha3_hash": sha3_hash,
            "vault_pointer": pointer,
            "integrity_tx_id": tx_integrity,
            "audit_tx_id": tx_audit,
            "block_height": self.ledger.block_height
        }

    def verify_resource_integrity(self, resource_id: str, pointer: str) -> Dict[str, Any]:
        """
        Live verification of off-chain encrypted EHR against immutable blockchain anchor.
        """
        try:
            decrypted_bytes = self.vault.retrieve_resource(pointer)
            is_valid, status_msg, ledger_hash = self.ledger.verify_fhir_hash(resource_id, decrypted_bytes)
            computed_hash = PostQuantumCryptoEngine.sha3_256_hash(decrypted_bytes)
            return {
                "resource_id": resource_id,
                "verified": is_valid,
                "status": status_msg,
                "computed_hash": computed_hash,
                "ledger_hash": ledger_hash
            }
        except Exception as e:
            return {
                "resource_id": resource_id,
                "verified": False,
                "status": f"TAMPERING_DETECTED_INTEGRITY_VIOLATION: {str(e)}",
                "computed_hash": "",
                "ledger_hash": ""
            }
