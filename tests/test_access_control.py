"""
tests/test_access_control.py
Tests for RBAC + ABAC policy engine (Phase 12).
Verifies normal, authorized, unauthorized, revoked, and emergency access modes.
"""

import pytest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.blockchain.chaincode import FabricChaincodeEngine
from backend.blockchain.access_control import AccessControlEngine, Role, PurposeOfUse

@pytest.fixture
def access_system():
    chaincode = FabricChaincodeEngine()
    engine = AccessControlEngine(chaincode)
    # Establish baseline consent
    chaincode.create_consent(
        patient_id="pt-access-01",
        authorized_org="Hospital_A",
        role=Role.DOCTOR.value,
        allowed_resources=["Observation", "Condition", "MedicationRequest"],
        purpose_of_use=PurposeOfUse.TREATMENT.value
    )
    return chaincode, engine

def test_authorized_access(access_system):
    _, engine = access_system
    decision = engine.evaluate_access(
        requestor_id="dr-smith-101",
        role=Role.DOCTOR,
        organization="Hospital_A",
        patient_id="pt-access-01",
        resource_type="Observation",
        purpose_of_use=PurposeOfUse.TREATMENT
    )
    assert decision["allowed"] is True
    assert decision["decision"] == "PERMIT"
    assert "audit_tx_id" in decision

def test_unauthorized_role_rbac(access_system):
    _, engine = access_system
    # Researcher attempting to access direct identifying Patient demographic
    decision = engine.evaluate_access(
        requestor_id="res-jones-501",
        role=Role.RESEARCHER,
        organization="Research_Organization",
        patient_id="pt-access-01",
        resource_type="Patient",
        purpose_of_use=PurposeOfUse.RESEARCH
    )
    assert decision["allowed"] is False
    assert decision["decision"] == "DENY_RBAC"

def test_unauthorized_missing_consent(access_system):
    _, engine = access_system
    # Doctor from unauthorized Hospital_B (no consent on file)
    decision = engine.evaluate_access(
        requestor_id="dr-rogue-202",
        role=Role.DOCTOR,
        organization="Hospital_B",
        patient_id="pt-access-01",
        resource_type="Observation",
        purpose_of_use=PurposeOfUse.TREATMENT
    )
    assert decision["allowed"] is False
    assert decision["decision"] == "DENY_CONSENT"

def test_revoked_consent_access(access_system):
    chaincode, engine = access_system
    # Find consent and revoke
    cid = list(chaincode.consent_state.keys())[0]
    chaincode.revoke_consent(cid, patient_id="pt-access-01")

    # Same request should now fail
    decision = engine.evaluate_access(
        requestor_id="dr-smith-101",
        role=Role.DOCTOR,
        organization="Hospital_A",
        patient_id="pt-access-01",
        resource_type="Observation",
        purpose_of_use=PurposeOfUse.TREATMENT
    )
    assert decision["allowed"] is False
    assert decision["decision"] == "DENY_CONSENT"
    assert "REVOKED" in decision["reason"]

def test_emergency_break_glass_access_valid(access_system):
    _, engine = access_system
    # Emergency provider without prior consent but valid medical justification
    decision = engine.evaluate_access(
        requestor_id="er-doc-999",
        role=Role.EMERGENCY_PROVIDER,
        organization="Hospital_B",
        patient_id="pt-access-01",
        resource_type="AllergyIntolerance",
        purpose_of_use=PurposeOfUse.EMERGENCY,
        is_emergency=True,
        emergency_justification="Patient unconscious in acute anaphylactic shock"
    )
    assert decision["allowed"] is True
    assert decision["decision"] == "PERMIT_EMERGENCY"
    assert decision["elevated_monitoring"] is True

def test_emergency_break_glass_access_invalid_no_justification(access_system):
    _, engine = access_system
    # Emergency access without justification must be rejected
    decision = engine.evaluate_access(
        requestor_id="er-doc-999",
        role=Role.EMERGENCY_PROVIDER,
        organization="Hospital_B",
        patient_id="pt-access-01",
        resource_type="AllergyIntolerance",
        purpose_of_use=PurposeOfUse.EMERGENCY,
        is_emergency=True,
        emergency_justification=None
    )
    assert decision["allowed"] is False
    assert decision["decision"] == "DENY"
