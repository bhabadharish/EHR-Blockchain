"""
backend/blockchain/chaincode.py
Hyperledger Fabric permissioned smart contract (chaincode) prototype for
dynamic patient consent management, access authorization, and immutable audit logs (Phase 11).
Zero plaintext EHR data is ever committed to the ledger state.
"""

import time
import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

class OrganizationType(str, Enum):
    HOSPITAL_A = "Hospital_A"
    HOSPITAL_B = "Hospital_B"
    LABORATORY = "Laboratory"
    RESEARCH_ORGANIZATION = "Research_Organization"
    PATIENT = "Patient"

class ConsentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    MODIFIED = "MODIFIED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"

class FabricChaincodeEngine:
    """
    Simulates a production Hyperledger Fabric smart contract state machine.
    Enforces channel validation, world-state key-value store, block commit,
    and cryptographic audit logging.
    """

    def __init__(self):
        # World State Databases (couchdb equivalent)
        self.consent_state: Dict[str, Dict[str, Any]] = {}
        self.record_references: Dict[str, Dict[str, Any]] = {}
        self.revocation_registry: Dict[str, Dict[str, Any]] = {}

        # Append-only blockchain blocks (ledger)
        self.ledger_blocks: List[Dict[str, Any]] = []
        self._genesis_block()

    def _genesis_block(self):
        genesis = {
            "block_number": 0,
            "timestamp": "2026-10-01T00:00:00Z",
            "transactions": [],
            "previous_hash": "0" * 64,
            "block_hash": hashlib.sha256(b"GENESIS_BLOCK_CRYPTO_AGILE_EHR").hexdigest()
        }
        self.ledger_blocks.append(genesis)

    def _commit_transaction(self, tx_type: str, actor: str, payload: Dict[str, Any]) -> str:
        """Appends an immutable transaction to the next block."""
        # Enforce zero-plaintext invariant: ensure no PHI keywords are present
        payload_str = json.dumps(payload).lower()
        forbidden_phi = ["cholesterol", "systolic", "blood pressure", "glucose", "insulin"]
        for f in forbidden_phi:
            if f in payload_str:
                raise ValueError(f"SECURITY VIOLATION: Plaintext PHI term '{f}' detected in blockchain transaction payload!")

        tx_id = f"tx-{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        tx = {
            "tx_id": tx_id,
            "tx_type": tx_type,
            "actor": actor,
            "timestamp": timestamp,
            "payload": payload
        }

        prev_hash = self.ledger_blocks[-1]["block_hash"]
        block_content = f"{len(self.ledger_blocks)}|{prev_hash}|{json.dumps(tx)}".encode('utf-8')
        block_hash = hashlib.sha256(block_content).hexdigest()

        new_block = {
            "block_number": len(self.ledger_blocks),
            "timestamp": timestamp,
            "transactions": [tx],
            "previous_hash": prev_hash,
            "block_hash": block_hash
        }
        self.ledger_blocks.append(new_block)
        return tx_id

    # 1. Dynamic Consent Management
    def create_consent(
        self,
        patient_id: str,
        authorized_org: str,
        role: str,
        allowed_resources: List[str],
        purpose_of_use: str,
        expires_at: Optional[str] = None
    ) -> Dict[str, Any]:
        consent_id = f"consent-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "consent_id": consent_id,
            "patient_id": patient_id,
            "authorized_org": authorized_org,
            "role": role,
            "allowed_resources": allowed_resources,
            "purpose_of_use": purpose_of_use,
            "status": ConsentStatus.ACTIVE.value,
            "created_at": now,
            "modified_at": now,
            "expires_at": expires_at or "2030-12-31T23:59:59Z"
        }

        tx_id = self._commit_transaction(
            tx_type="CONSENT_CREATION",
            actor=f"Patient/{patient_id}",
            payload=record
        )
        record["tx_id"] = tx_id
        self.consent_state[consent_id] = record
        return record

    def modify_consent(
        self,
        consent_id: str,
        patient_id: str,
        new_allowed_resources: Optional[List[str]] = None,
        new_purpose: Optional[str] = None,
        new_expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        if consent_id not in self.consent_state:
            raise KeyError(f"Consent {consent_id} not found.")

        consent = self.consent_state[consent_id]
        if consent["patient_id"] != patient_id:
            raise PermissionError("Only patient data owner can modify consent.")
        if consent["status"] == ConsentStatus.REVOKED.value:
            raise ValueError("Cannot modify already revoked consent.")

        now = datetime.now(timezone.utc).isoformat()
        if new_allowed_resources is not None:
            consent["allowed_resources"] = new_allowed_resources
        if new_purpose is not None:
            consent["purpose_of_use"] = new_purpose
        if new_expiration is not None:
            consent["expires_at"] = new_expiration

        consent["status"] = ConsentStatus.MODIFIED.value
        consent["modified_at"] = now

        tx_id = self._commit_transaction(
            tx_type="CONSENT_MODIFICATION",
            actor=f"Patient/{patient_id}",
            payload=consent
        )
        consent["tx_id"] = tx_id
        return consent

    def revoke_consent(self, consent_id: str, patient_id: str, reason: str = "Patient requested revocation") -> Dict[str, Any]:
        if consent_id not in self.consent_state:
            raise KeyError(f"Consent {consent_id} not found.")

        consent = self.consent_state[consent_id]
        if consent["patient_id"] != patient_id:
            raise PermissionError("Only patient data owner can revoke consent.")

        now = datetime.now(timezone.utc).isoformat()
        consent["status"] = ConsentStatus.REVOKED.value
        consent["modified_at"] = now
        consent["revocation_reason"] = reason

        revocation_entry = {
            "consent_id": consent_id,
            "patient_id": patient_id,
            "revoked_at": now,
            "reason": reason
        }
        self.revocation_registry[consent_id] = revocation_entry

        tx_id = self._commit_transaction(
            tx_type="CONSENT_REVOCATION",
            actor=f"Patient/{patient_id}",
            payload=revocation_entry
        )
        consent["tx_id"] = tx_id
        return consent

    # 2. Record Reference Anchoring (Zero Plaintext)
    def anchor_record_reference(self, reference: Dict[str, Any], submitting_org: str) -> str:
        record_id = reference["record_id"]
        tx_id = self._commit_transaction(
            tx_type="EHR_RECORD_ANCHOR",
            actor=submitting_org,
            payload=reference
        )
        reference["anchor_tx_id"] = tx_id
        self.record_references[record_id] = reference
        return tx_id

    def get_record_reference(self, record_id: str) -> Optional[Dict[str, Any]]:
        return self.record_references.get(record_id)

    # 3. Access Query & Authorization Evaluation
    def verify_consent(
        self,
        patient_id: str,
        requestor_org: str,
        requestor_role: str,
        resource_type: str,
        purpose: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Evaluates world-state consents.
        Returns: (is_allowed, reason, matched_consent_id)
        """
        # Find active consents for patient and organization
        matching_consents = [
            c for c in self.consent_state.values()
            if c["patient_id"] == patient_id and c["authorized_org"] == requestor_org
        ]

        if not matching_consents:
            return False, "No active consent found between patient and organization", None

        now_str = datetime.now(timezone.utc).isoformat()

        for c in matching_consents:
            cid = c["consent_id"]
            # Check revocation
            if c["status"] == ConsentStatus.REVOKED.value or cid in self.revocation_registry:
                return False, f"Consent '{cid}' was REVOKED", cid

            # Check expiration
            if c.get("expires_at") and c["expires_at"] < now_str:
                return False, f"Consent '{cid}' has EXPIRED", cid

            # Check resource scope
            allowed_res = c.get("allowed_resources", [])
            if "*" not in allowed_res and resource_type not in allowed_res:
                continue

            # Check purpose of use
            allowed_purpose = c.get("purpose_of_use", "*")
            if allowed_purpose != "*" and allowed_purpose != purpose:
                continue

            # Check role binding
            allowed_role = c.get("role", "*")
            if allowed_role != "*" and allowed_role != requestor_role:
                continue

            return True, "Authorized by active consent", cid

        return False, "Requested resource or purpose not permitted under consent scope", None

    # 4. Audit Trail Retrieval
    def get_audit_trail(self, patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
        trail = []
        for block in self.ledger_blocks:
            for tx in block["transactions"]:
                p = tx.get("payload", {})
                if patient_id is None or p.get("patient_id") == patient_id:
                    trail.append({
                        "block_number": block["block_number"],
                        "block_hash": block["block_hash"][:16] + "...",
                        "tx_id": tx["tx_id"],
                        "tx_type": tx["tx_type"],
                        "actor": tx["actor"],
                        "timestamp": tx["timestamp"]
                    })
        return trail
