"""
tests/test_blockchain.py
Tests for Hyperledger Fabric chaincode block chaining, zero-plaintext enforcement,
and tamper-evident audit trails (Phase 11).
"""

import pytest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.blockchain.chaincode import FabricChaincodeEngine

def test_blockchain_block_commit_and_chaining():
    engine = FabricChaincodeEngine()
    assert len(engine.ledger_blocks) == 1  # Genesis block

    # Commit transactions
    c1 = engine.create_consent("pt-1", "Hospital_A", "DOCTOR", ["*"], "TREATMENT")
    c2 = engine.create_consent("pt-2", "Hospital_B", "DOCTOR", ["*"], "TREATMENT")

    assert len(engine.ledger_blocks) == 3

    # Verify cryptographic block hash linkage
    for i in range(1, len(engine.ledger_blocks)):
        curr_block = engine.ledger_blocks[i]
        prev_block = engine.ledger_blocks[i - 1]
        assert curr_block["previous_hash"] == prev_block["block_hash"]
        assert len(curr_block["block_hash"]) == 64

def test_zero_plaintext_phi_invariant():
    engine = FabricChaincodeEngine()
    # Attempting to commit clinical plaintext PHI must raise security violation
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        engine._commit_transaction(
            tx_type="MALICIOUS_PHI_TX",
            actor="Rogue_Node",
            payload={"patient_id": "pt-1", "diagnosis": "Severe glucose intolerance and high systolic reading"}
        )

def test_audit_trail_retrieval():
    engine = FabricChaincodeEngine()
    engine.create_consent("pt-audit-1", "Hospital_A", "DOCTOR", ["*"], "TREATMENT")
    engine.create_consent("pt-audit-2", "Hospital_B", "DOCTOR", ["*"], "TREATMENT")

    trail_pt1 = engine.get_audit_trail(patient_id="pt-audit-1")
    assert len(trail_pt1) == 1
    assert trail_pt1[0]["actor"] == "Patient/pt-audit-1"

    trail_all = engine.get_audit_trail()
    assert len(trail_all) == 2
