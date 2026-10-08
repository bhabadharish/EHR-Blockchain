import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard.inference.model_loader import FrozenModelLoader

def main():
    print("=" * 60)
    print("TEST: INFERENCE PIPELINE (OFFLINE FROZEN MODEL)")
    print("=" * 60)

    loader = FrozenModelLoader()
    print("1. Integrity Verification:")
    print(f"   Model Status:        {loader.integrity_status['model_status']}")
    print(f"   Preprocessor Status: {loader.integrity_status['preprocessor_status']}")
    print(f"   Optimal Threshold:   {loader.optimal_threshold}")
    assert loader.integrity_status['model_status'] == 'VERIFIED'
    assert loader.integrity_status['preprocessor_status'] == 'VERIFIED'

    # Test Sample 1: Benign Clinical Routine
    benign_sample = {
        "flow_duration": 0.04, "packet_count": 10, "byte_count": 1200,
        "packet_rate": 250.0, "byte_rate": 30000.0, "dst_port": 443, "protocol": 6,
        "resource_sensitivity": 0.5, "auth_status": 1, "failed_auth_count": 0,
        "request_frequency": 2.0, "burst_score": 0.05, "historical_risk": 0.02,
        "user_role": "doctor", "resource_type": "Observation", "operation": "read"
    }

    res_benign = loader.predict(benign_sample)
    print("\n2. Benign Routine Inference:")
    print(f"   Predicted Class:    {res_benign['predicted_class']}")
    print(f"   Threat Probability: {res_benign['threat_probability']:.4f}")
    print(f"   Decision:           {res_benign['decision']}")
    print(f"   Risk Level:         {res_benign['risk_level']}")
    print(f"   Latency:            {res_benign['latency_ms']:.2f} ms")
    assert res_benign['decision'] in ["ALLOW", "REVIEW"]

    # Test Sample 2: Malicious DDoS Attack
    attack_sample = {
        "flow_duration": 0.01, "packet_count": 500, "byte_count": 120000,
        "packet_rate": 50000.0, "byte_rate": 12000000.0, "dst_port": 443, "protocol": 6,
        "resource_sensitivity": 0.8, "auth_status": 0, "failed_auth_count": 15,
        "request_frequency": 120.0, "burst_score": 0.98, "historical_risk": 0.95,
        "user_role": "iomt_device", "resource_type": "Device", "operation": "read"
    }

    res_attack = loader.predict(attack_sample)
    print("\n3. Attack Injection Inference:")
    print(f"   Predicted Class:    {res_attack['predicted_class']}")
    print(f"   Threat Probability: {res_attack['threat_probability']:.4f}")
    print(f"   Decision:           {res_attack['decision']}")
    print(f"   Risk Level:         {res_attack['risk_level']}")
    print(f"   Latency:            {res_attack['latency_ms']:.2f} ms")
    assert res_attack['decision'] == "BLOCK"
    assert res_attack['predicted_class'] == "Cyber Threat"

    print("\n✓ Inference pipeline verified successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
