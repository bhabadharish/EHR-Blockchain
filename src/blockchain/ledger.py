import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple

class HyperledgerFabricBlock:
    def __init__(self, index: int, previous_hash: str, transactions: List[Dict[str, Any]], timestamp: Optional[float] = None):
        self.index = index
        self.timestamp = timestamp or time.time()
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.block_hash = self.compute_block_hash()

    def compute_block_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha3_256(block_string.encode()).hexdigest()

class FabricPermissionedLedger:
    """
    Simulated High-Fidelity Hyperledger Fabric Permissioned Ledger.
    Supports Consortium Channel with 4 Organizations:
      - Org1: Hospital_A
      - Org2: Hospital_B
      - Org3: Hospital_C
      - Org4: Research_Consortium
    Maintains:
      - Blockchain (Ordered sequence of immutable blocks)
      - World State (Key-Value Store)
      - Chaincode Subsystems: Consent, Access, Audit, Integrity, Threat
    """

    ORGANIZATIONS = ["Hospital_A", "Hospital_B", "Hospital_C", "Research_Consortium"]

    def __init__(self):
        self.chain: List[HyperledgerFabricBlock] = []
        self.world_state: Dict[str, Any] = {}
        self.pending_transactions: List[Dict[str, Any]] = []
        self.block_height = 0
        
        # Initialize Genesis Block
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis_tx = [{
            "tx_id": "genesis_tx_0000000000000000",
            "type": "CONFIG",
            "channel": "healthcare-consortium-channel",
            "organizations": self.ORGANIZATIONS,
            "chaincodes": ["ConsentCC", "AccessCC", "AuditCC", "IntegrityCC", "ThreatCC"],
            "timestamp": 1704067200.0 # 2024-01-01
        }]
        genesis_block = HyperledgerFabricBlock(
            index=0,
            previous_hash="0" * 64,
            transactions=genesis_tx,
            timestamp=1704067200.0
        )
        self.chain.append(genesis_block)
        self.block_height = 1

    def commit_transaction(self, tx: Dict[str, Any]) -> str:
        """
        Commits endorsed transaction to ledger and updates World State.
        """
        tx["timestamp"] = tx.get("timestamp", time.time())
        tx["tx_id"] = hashlib.sha3_256(f"{tx['type']}_{time.time()}_{len(self.chain)}".encode()).hexdigest()[:32]
        
        # Update World State
        if "state_key" in tx and "state_value" in tx:
            self.world_state[tx["state_key"]] = tx["state_value"]
            
        self.pending_transactions.append(tx)
        
        # Automatic batching into block (simulating Fabric Raft orderer block-cutting)
        if len(self.pending_transactions) >= 1:
            prev_hash = self.chain[-1].block_hash
            new_block = HyperledgerFabricBlock(
                index=self.block_height,
                previous_hash=prev_hash,
                transactions=list(self.pending_transactions)
            )
            self.chain.append(new_block)
            self.block_height += 1
            self.pending_transactions = []
            
        return tx["tx_id"]

    # -------------------------------------------------------------
    # 1. Consent Chaincode
    # -------------------------------------------------------------
    def create_consent(self, consent_id: str, patient_id: str, grantee: str, allowed_resources: List[str], expires_at: float) -> str:
        state_key = f"consent:{consent_id}"
        state_value = {
            "consent_id": consent_id,
            "patient_id": patient_id,
            "grantee": grantee,
            "allowed_resources": allowed_resources,
            "status": "ACTIVE",
            "expires_at": expires_at
        }
        return self.commit_transaction({
            "chaincode": "ConsentCC",
            "type": "createConsent",
            "state_key": state_key,
            "state_value": state_value
        })

    def revoke_consent(self, consent_id: str) -> str:
        state_key = f"consent:{consent_id}"
        if state_key in self.world_state:
            self.world_state[state_key]["status"] = "REVOKED"
        return self.commit_transaction({
            "chaincode": "ConsentCC",
            "type": "revokeConsent",
            "state_key": state_key,
            "state_value": self.world_state.get(state_key, {})
        })

    def check_consent(self, patient_id: str, grantee: str, resource_type: str) -> Tuple[bool, str]:
        for k, v in self.world_state.items():
            if k.startswith("consent:") and v.get("patient_id") == patient_id and v.get("grantee") == grantee:
                if v.get("status") != "ACTIVE":
                    return False, "Consent has been revoked."
                if time.time() > v.get("expires_at", 0):
                    return False, "Consent has expired."
                if resource_type in v.get("allowed_resources", []) or "*" in v.get("allowed_resources", []):
                    return True, "Valid active consent found."
        return False, "No active consent agreement found on ledger."

    # -------------------------------------------------------------
    # 2. Access Chaincode (RBAC / ABAC)
    # -------------------------------------------------------------
    def register_actor(self, actor_id: str, role: str, org: str, clearance: str = "CONFIDENTIAL") -> str:
        state_key = f"actor:{actor_id}"
        state_value = {
            "actor_id": actor_id,
            "role": role,
            "organization": org,
            "clearance": clearance,
            "status": "ACTIVE"
        }
        return self.commit_transaction({
            "chaincode": "AccessCC",
            "type": "registerActor",
            "state_key": state_key,
            "state_value": state_value
        })

    def check_access(self, actor_id: str, resource_type: str, operation: str) -> Tuple[bool, str]:
        actor = self.world_state.get(f"actor:{actor_id}")
        if not actor or actor.get("status") != "ACTIVE":
            return False, f"Actor {actor_id} is unregistered or suspended."
        
        role = actor.get("role", "")
        # Role matrix
        role_permissions = {
            "doctor": ["read", "vread", "search", "create", "update"],
            "nurse": ["read", "vread", "search", "create"],
            "admin": ["read", "vread", "search", "create", "update", "delete"],
            "researcher": ["read", "search"],
            "patient": ["read", "vread"],
            "lab_tech": ["read", "create", "update"],
            "iomt_device": ["create", "update"]
        }
        allowed_ops = role_permissions.get(role, [])
        if operation not in allowed_ops:
            return False, f"Role '{role}' lacks permission for operation '{operation}'."
        return True, f"Role '{role}' authorized for '{operation}'."

    # -------------------------------------------------------------
    # 3. Audit Chaincode
    # -------------------------------------------------------------
    def record_audit_event(self, actor_id: str, action: str, resource_id: str, outcome: str, threat_score: float) -> str:
        event_data = {
            "actor_id": actor_id,
            "action": action,
            "resource_id": resource_id,
            "outcome": outcome,
            "threat_score": threat_score,
            "timestamp": time.time()
        }
        return self.commit_transaction({
            "chaincode": "AuditCC",
            "type": "recordAuditEvent",
            "event_data": event_data
        })

    # -------------------------------------------------------------
    # 4. Integrity Chaincode (Off-Chain Verification)
    # -------------------------------------------------------------
    def register_fhir_hash(self, resource_id: str, sha3_hash: str, encrypted_pointer: str) -> str:
        state_key = f"integrity:{resource_id}"
        state_value = {
            "resource_id": resource_id,
            "sha3_hash": sha3_hash,
            "encrypted_pointer": encrypted_pointer,
            "timestamp": time.time()
        }
        return self.commit_transaction({
            "chaincode": "IntegrityCC",
            "type": "registerFHIRHash",
            "state_key": state_key,
            "state_value": state_value
        })

    def verify_fhir_hash(self, resource_id: str, current_content_bytes: bytes) -> Tuple[bool, str, str]:
        state_key = f"integrity:{resource_id}"
        registered = self.world_state.get(state_key)
        if not registered:
            return False, "Not registered on blockchain", ""
        
        current_hash = hashlib.sha3_256(current_content_bytes).hexdigest()
        ledger_hash = registered.get("sha3_hash", "")
        is_valid = (current_hash == ledger_hash)
        status_msg = "INTEGRITY_VERIFIED_MATCH" if is_valid else "TAMPERING_DETECTED_MISMATCH"
        return is_valid, status_msg, ledger_hash

    # -------------------------------------------------------------
    # 5. Threat Chaincode
    # -------------------------------------------------------------
    def record_threat(self, actor_or_device_id: str, threat_category: str, confidence: float, recommended_action: str) -> str:
        state_key = f"threat:{actor_or_device_id}"
        state_value = {
            "id": actor_or_device_id,
            "category": threat_category,
            "confidence": confidence,
            "action": recommended_action,
            "status": "QUARANTINED" if "QUARANTINE" in recommended_action or "BLOCK" in recommended_action else "MONITORED",
            "updated_at": time.time()
        }
        return self.commit_transaction({
            "chaincode": "ThreatCC",
            "type": "recordThreat",
            "state_key": state_key,
            "state_value": state_value
        })
