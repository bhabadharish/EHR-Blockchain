"""
tests/test_consent.py
Tests for dynamic patient consent creation, modification, revocation,
expiration, and multi-organization enforcement (Phase 11).
"""

import pytest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.blockchain.chaincode import FabricChaincodeEngine, ConsentStatus

@pytest.fixture
def chaincode():
    return FabricChaincodeEngine()

def test_consent_lifecycle(chaincode):
    # 1. Create consent
    consent = chaincode.create_consent(
        patient_id="pt-1001",
        authorized_org="Hospital_B",
        role="DOCTOR",
        allowed_resources=["Observation", "Condition"],
        purpose_of_use="TREATMENT"
    )
    cid = consent["consent_id"]
    assert consent["status"] == ConsentStatus.ACTIVE.value
    assert consent["patient_id"] == "pt-1001"

    # Verify query
    ok, reason, matched = chaincode.verify_consent(
        patient_id="pt-1001",
        requestor_org="Hospital_B",
        requestor_role="DOCTOR",
        resource_type="Observation",
        purpose="TREATMENT"
    )
    assert ok is True
    assert matched == cid

    # 2. Modify consent (add MedicationRequest)
    modified = chaincode.modify_consent(
        consent_id=cid,
        patient_id="pt-1001",
        new_allowed_resources=["Observation", "Condition", "MedicationRequest"]
    )
    assert modified["status"] == ConsentStatus.MODIFIED.value
    assert "MedicationRequest" in modified["allowed_resources"]

    ok_med, _, _ = chaincode.verify_consent(
        patient_id="pt-1001",
        requestor_org="Hospital_B",
        requestor_role="DOCTOR",
        resource_type="MedicationRequest",
        purpose="TREATMENT"
    )
    assert ok_med is True

    # 3. Revoke consent
    revoked = chaincode.revoke_consent(
        consent_id=cid,
        patient_id="pt-1001",
        reason="Patient discontinued care at Hospital B"
    )
    assert revoked["status"] == ConsentStatus.REVOKED.value

    # Subsequent access must be blocked
    ok_revoked, reason_rev, _ = chaincode.verify_consent(
        patient_id="pt-1001",
        requestor_org="Hospital_B",
        requestor_role="DOCTOR",
        resource_type="Observation",
        purpose="TREATMENT"
    )
    assert ok_revoked is False
    assert "REVOKED" in reason_rev

def test_consent_unauthorized_scope(chaincode):
    chaincode.create_consent(
        patient_id="pt-1002",
        authorized_org="Research_Organization",
        role="RESEARCHER",
        allowed_resources=["Observation"],
        purpose_of_use="RESEARCH"
    )
    # Requesting Condition should fail
    ok, reason, _ = chaincode.verify_consent(
        patient_id="pt-1002",
        requestor_org="Research_Organization",
        requestor_role="RESEARCHER",
        resource_type="Condition",
        purpose="RESEARCH"
    )
    assert ok is False
