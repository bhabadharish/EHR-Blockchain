"""
backend/blockchain/access_control.py
Role-Based and Attribute-Based Access Control (RBAC + ABAC) engine (Phase 12).
Evaluates contextual requests against organizational identity, user role,
environmental attributes, purpose of use, and blockchain consent status.
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from backend.blockchain.chaincode import FabricChaincodeEngine, ConsentStatus

class Role(str, Enum):
    PATIENT = "PATIENT"
    DOCTOR = "DOCTOR"
    NURSE = "NURSE"
    LAB = "LAB"
    RESEARCHER = "RESEARCHER"
    ADMIN = "ADMIN"
    EMERGENCY_PROVIDER = "EMERGENCY_PROVIDER"

class PurposeOfUse(str, Enum):
    TREATMENT = "TREATMENT"
    RESEARCH = "RESEARCH"
    EMERGENCY = "EMERGENCY"
    AUDIT = "AUDIT"
    PAYMENT = "PAYMENT"

# RBAC Matrix: Baseline permitted resource categories per role
ROLE_PERMISSIONS: Dict[Role, list] = {
    Role.PATIENT: ["*"],
    Role.DOCTOR: ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "Procedure", "DiagnosticReport", "AllergyIntolerance"],
    Role.NURSE: ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "AllergyIntolerance"],
    Role.LAB: ["Patient", "Observation", "DiagnosticReport"],
    Role.RESEARCHER: ["Condition", "Observation", "MedicationRequest", "Procedure"],  # Direct Patient Demographics restricted
    Role.ADMIN: ["Organization", "Practitioner"],
    Role.EMERGENCY_PROVIDER: ["Patient", "AllergyIntolerance", "Condition", "Observation", "MedicationRequest"]
}

class AccessControlEngine:
    """
    Unified RBAC + ABAC decision engine coordinating with the blockchain chaincode.
    """

    def __init__(self, chaincode: FabricChaincodeEngine):
        self.chaincode = chaincode

    def evaluate_access(
        self,
        requestor_id: str,
        role: Role,
        organization: str,
        patient_id: str,
        resource_type: str,
        purpose_of_use: PurposeOfUse,
        is_emergency: bool = False,
        emergency_justification: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates contextual access.
        Returns detailed decision packet: {allowed, decision, reason, consent_id, audit_tx_id}
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Emergency "Break-Glass" Access Override
        if is_emergency or role == Role.EMERGENCY_PROVIDER or purpose_of_use == PurposeOfUse.EMERGENCY:
            if not emergency_justification:
                return {
                    "allowed": False,
                    "decision": "DENY",
                    "reason": "Emergency break-glass access requires documented clinical justification.",
                    "audit_flag": "FAILED_EMERGENCY_ACCESS"
                }
            
            # Log high-priority immutable audit event
            tx_id = self.chaincode._commit_transaction(
                tx_type="EMERGENCY_BREAK_GLASS_ACCESS",
                actor=f"{role.value}/{requestor_id}@{organization}",
                payload={
                    "patient_id": patient_id,
                    "resource_type": resource_type,
                    "justification": emergency_justification,
                    "elevated_audit": True
                }
            )
            return {
                "allowed": True,
                "decision": "PERMIT_EMERGENCY",
                "reason": "Authorized via Emergency Break-Glass protocol.",
                "audit_tx_id": tx_id,
                "elevated_monitoring": True
            }

        # 2. RBAC Verification
        allowed_res_types = ROLE_PERMISSIONS.get(role, [])
        if "*" not in allowed_res_types and resource_type not in allowed_res_types:
            tx_id = self.chaincode._commit_transaction(
                tx_type="ACCESS_DENIED_RBAC",
                actor=f"{role.value}/{requestor_id}@{organization}",
                payload={"patient_id": patient_id, "resource_type": resource_type, "role": role.value}
            )
            return {
                "allowed": False,
                "decision": "DENY_RBAC",
                "reason": f"Role '{role.value}' lacks clearance to access '{resource_type}' resources.",
                "audit_tx_id": tx_id
            }

        # 3. Patient self-access
        if role == Role.PATIENT and requestor_id == patient_id:
            tx_id = self.chaincode._commit_transaction(
                tx_type="PATIENT_SELF_ACCESS",
                actor=f"Patient/{patient_id}",
                payload={"patient_id": patient_id, "resource_type": resource_type}
            )
            return {
                "allowed": True,
                "decision": "PERMIT_DATA_OWNER",
                "reason": "Patient authenticated as record owner.",
                "audit_tx_id": tx_id
            }

        # 4. ABAC & Dynamic Blockchain Consent Verification
        has_consent, consent_reason, consent_id = self.chaincode.verify_consent(
            patient_id=patient_id,
            requestor_org=organization,
            requestor_role=role.value,
            resource_type=resource_type,
            purpose=purpose_of_use.value
        )

        if not has_consent:
            tx_id = self.chaincode._commit_transaction(
                tx_type="ACCESS_DENIED_CONSENT",
                actor=f"{role.value}/{requestor_id}@{organization}",
                payload={"patient_id": patient_id, "resource_type": resource_type, "reason": consent_reason}
            )
            return {
                "allowed": False,
                "decision": "DENY_CONSENT",
                "reason": f"Access denied: {consent_reason}.",
                "consent_id": consent_id,
                "audit_tx_id": tx_id
            }

        # 5. Full Authorization Permit
        tx_id = self.chaincode._commit_transaction(
            tx_type="ACCESS_PERMITTED",
            actor=f"{role.value}/{requestor_id}@{organization}",
            payload={"patient_id": patient_id, "resource_type": resource_type, "consent_id": consent_id}
        )
        return {
            "allowed": True,
            "decision": "PERMIT",
            "reason": f"Authorized under active blockchain consent '{consent_id}'.",
            "consent_id": consent_id,
            "audit_tx_id": tx_id
        }
