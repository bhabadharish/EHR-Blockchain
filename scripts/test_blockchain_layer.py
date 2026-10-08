import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.blockchain.client import FabricEHRClient

def main():
    print("=" * 60)
    print("TEST: HYPERLEDGER FABRIC CONSORTIUM BLOCKCHAIN & VAULT")
    print("=" * 60)

    client = FabricEHRClient()
    print("1. Ledger Initialization:")
    print(f"   Initial Block Height: {client.ledger.block_height}")
    print(f"   Consortium Orgs:      {client.ledger.ORGANIZATIONS}")

    # 2. Register EHR Exchange Transaction
    print("\n2. Anchoring FHIR Transaction to Ledger:")
    fhir_sample = {
        "resourceType": "Observation",
        "id": "obs-test-99",
        "status": "final",
        "code": "8867-4",
        "value": 72.0,
        "unit": "beats/min"
    }

    reg_res = client.register_ehr_exchange(
        actor_id="dr_carol",
        patient_id="pt-1024",
        resource_id="obs-test-99",
        fhir_resource=fhir_sample,
        threat_score=0.12
    )

    print(f"   Integrity Tx ID:  {reg_res['integrity_tx_id']}")
    print(f"   Audit Tx ID:      {reg_res['audit_tx_id']}")
    print(f"   Block Height:     {reg_res['block_height']}")
    print(f"   Ledger SHA3 Hash: {reg_res['sha3_hash']}")
    print(f"   Vault Pointer:    {reg_res['vault_pointer']}")
    assert os.path.exists(reg_res['vault_pointer']), "Off-chain vault file was not created!"

    # 3. Verify Resource Integrity (Genuine Unaltered Record)
    print("\n3. Testing Resource Integrity Verification (Unaltered):")
    v_clean = client.verify_resource_integrity("obs-test-99", reg_res["vault_pointer"])
    print(f"   Status:           {v_clean['status']}")
    print(f"   Verified:         {v_clean['verified']}")
    print(f"   Computed Hash:    {v_clean['computed_hash']}")
    print(f"   Ledger Hash:      {v_clean['ledger_hash']}")
    assert v_clean["verified"] is True
    print("   Unaltered Record Integrity: VERIFIED ✓")

    # 4. Tamper Simulation (Mutate Bytes on Disk)
    print("\n4. Simulating Malicious Tampering in Off-Chain Vault:")
    tamper_obs = {"resourceType": "Observation", "id": "obs-tamper-target", "value": 120.0}
    reg_tamper = client.register_ehr_exchange("dr_attacker", "pt-88", "obs-tamper-target", tamper_obs, 0.45)
    
    # Mutate byte on disk
    with open(reg_tamper["vault_pointer"], "r+b") as f:
        f.seek(16)
        f.write(b"\xde\xad\xbe\xef")

    v_tampered = client.verify_resource_integrity("obs-tamper-target", reg_tamper["vault_pointer"])
    print(f"   Status:           {v_tampered['status']}")
    print(f"   Verified:         {v_tampered['verified']}")
    print(f"   Computed Hash:    {v_tampered['computed_hash']}")
    print(f"   Ledger Hash:      {v_tampered['ledger_hash']}")
    assert v_tampered["verified"] is False
    print("   Tamper Detection Alert: VERIFIED ✓")

    print("\n✓ Blockchain consortium audit and tamper detection successfully verified!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
