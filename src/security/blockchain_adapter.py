"""Permissioned Blockchain Audit Ledger Adapter for HAB-IDS (Phase 24).

Implements hybrid on-chain / off-chain architecture:
- On-chain: Lightweight cryptographic digest (FHIR hash, event ID, timestamp,
  threat classification, ML model version, digital signature, block merkle root).
- Off-chain: AES-256-GCM encrypted clinical payload in local vault.
Decoupled benchmarking: transaction latency, throughput, and verification time
are benchmarked separately without affecting ML intrusion detection metrics.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
import numpy as np


class BlockchainAuditAdapter:
    """Decoupled permissioned ledger adapter with cryptographic verification."""

    def __init__(self, vault_dir: str = "data/processed/offchain_vault"):
        self.vault_dir = vault_dir
        os.makedirs(self.vault_dir, exist_ok=True)
        self.chain: List[Dict[str, Any]] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis_block = {
            "block_index": 0,
            "timestamp": 1704067200.0,
            "transactions": [],
            "previous_hash": "0" * 64,
            "merkle_root": "0" * 64,
            "block_hash": hashlib.sha256(b"genesis_block_hab_ids").hexdigest(),
        }
        self.chain.append(genesis_block)

    def record_security_audit(
        self,
        event_id: str,
        fhir_resource_id: str,
        resource_hash: str,
        risk_level: str,
        model_version: str,
        prediction: int,
        digital_signature: str,
        encrypted_payload: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Record off-chain encrypted payload and commit on-chain audit receipt."""
        start_t = time.perf_counter()

        # 1. Off-chain vault storage
        if encrypted_payload:
            payload_path = os.path.join(self.vault_dir, f"{fhir_resource_id}.enc")
            with open(payload_path, "wb") as f:
                f.write(encrypted_payload)

        # 2. On-chain transaction payload
        tx = {
            "event_id": event_id,
            "timestamp": time.time(),
            "resource_id": fhir_resource_id,
            "resource_sha256": resource_hash,
            "risk_level": risk_level,
            "model_version": model_version,
            "prediction": prediction,
            "signature": digital_signature[:32] + "...",
        }

        # 3. Commit to micro-block
        prev_block = self.chain[-1]
        block_idx = len(self.chain)
        tx_bytes = json.dumps(tx, sort_keys=True).encode()
        merkle_root = hashlib.sha256(tx_bytes).hexdigest()
        block_header = f"{block_idx}_{prev_block['block_hash']}_{merkle_root}".encode()
        block_hash = hashlib.sha256(block_header).hexdigest()

        block = {
            "block_index": block_idx,
            "timestamp": tx["timestamp"],
            "transactions": [tx],
            "previous_hash": prev_block["block_hash"],
            "merkle_root": merkle_root,
            "block_hash": block_hash,
        }
        self.chain.append(block)
        commit_latency_ms = (time.perf_counter() - start_t) * 1000

        return {
            "block_index": block_idx,
            "block_hash": block_hash,
            "commit_latency_ms": round(commit_latency_ms, 3),
            "status": "COMMITTED"
        }

    def verify_audit_integrity(self, block_index: int) -> bool:
        """Verify on-chain cryptographic link and merkle integrity."""
        if block_index <= 0 or block_index >= len(self.chain):
            return False
        curr_b = self.chain[block_index]
        prev_b = self.chain[block_index - 1]
        
        # Check parent hash pointer
        if curr_b["previous_hash"] != prev_b["block_hash"]:
            return False

        # Check merkle root
        tx_bytes = json.dumps(curr_b["transactions"][0], sort_keys=True).encode()
        expected_root = hashlib.sha256(tx_bytes).hexdigest()
        if curr_b["merkle_root"] != expected_root:
            return False

        return True


def benchmark_blockchain_adapter(n_tx: int = 500) -> Dict[str, Any]:
    """Benchmark commit latency, throughput (TPS), and verification time."""
    adapter = BlockchainAuditAdapter()
    dummy_payload = b"encrypted_clinical_observation_sample"
    latencies = []

    t_start = time.perf_counter()
    for i in range(n_tx):
        t0 = time.perf_counter()
        adapter.record_security_audit(
            event_id=f"evt_{i}",
            fhir_resource_id=f"res_{i}",
            resource_hash=hashlib.sha256(f"sample_{i}".encode()).hexdigest(),
            risk_level="CRITICAL" if i % 4 == 0 else "LOW",
            model_version="1.0.0-HAB-IDS",
            prediction=1 if i % 4 == 0 else 0,
            digital_signature="dsa_mock_signature_hash_bytes",
            encrypted_payload=dummy_payload
        )
        latencies.append((time.perf_counter() - t0) * 1000)

    total_time = time.perf_counter() - t_start
    throughput_tps = round(n_tx / max(total_time, 1e-4), 1)

    # Verification latency
    t_ver = time.perf_counter()
    for i in range(1, min(n_tx, 100)):
        assert adapter.verify_audit_integrity(i)
    ver_time_ms = ((time.perf_counter() - t_ver) / min(n_tx, 100)) * 1000

    return {
        "Total_Transactions": n_tx,
        "Throughput_TPS": throughput_tps,
        "Commit_Latency_Mean_ms": round(float(np.mean(latencies)), 3),
        "Commit_Latency_P95_ms": round(float(np.percentile(latencies, 95)), 3),
        "Audit_Verification_Time_per_Block_ms": round(float(ver_time_ms), 4),
        "OnChain_Block_Count": len(adapter.chain),
    }
