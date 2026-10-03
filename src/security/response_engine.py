import os
import time
import json
from typing import Dict, Any, Tuple

class ThreatAwareResponseEngine:
    """
    Zero-Trust Adaptive Threat Response Engine.
    Evaluates multi-factor context:
      - Threat Probability (Model inference output)
      - FHIR Resource Sensitivity (Clinical severity / HIPAA tier)
      - User Role & Clearance Level
      - Device Trust & Behavioral Drift
      - Active Consent Agreement
      - Historical Risk Index
    """

    ROLE_TRUST = {
        "doctor": 0.85,
        "nurse": 0.80,
        "admin": 0.90,
        "researcher": 0.65,
        "lab_tech": 0.75,
        "patient": 0.70,
        "iomt_device": 0.80
    }

    DEVICE_TRUST = {
        "clinical_workstation": 0.90,
        "icu_monitor": 0.95,
        "infusion_pump": 0.95,
        "mobile_tablet": 0.70,
        "wearable_sensor": 0.65,
        "external_api_client": 0.50
    }

    def __init__(self, log_path: str = "results/security_decisions.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def calculate_contextual_risk(
        self,
        threat_probability: float,
        resource_sensitivity: float,
        user_role: str,
        device_type: str,
        historical_risk: float = 0.1,
        has_consent: bool = True
    ) -> float:
        role_score = 1.0 - self.ROLE_TRUST.get(user_role, 0.50)
        device_score = 1.0 - self.DEVICE_TRUST.get(device_type, 0.50)
        consent_penalty = 0.0 if has_consent else 0.40

        # Weighted multi-factor contextual risk formula
        risk = (
            0.45 * threat_probability +
            0.20 * resource_sensitivity +
            0.15 * historical_risk +
            0.10 * role_score +
            0.10 * device_score +
            consent_penalty
        )
        return float(min(max(risk, 0.0), 1.0))

    def evaluate_request(
        self,
        actor_id: str,
        user_role: str,
        device_id: str,
        device_type: str,
        resource_type: str,
        resource_sensitivity: float,
        operation: str,
        threat_probability: float,
        has_consent: bool = True,
        historical_risk: float = 0.1
    ) -> Dict[str, Any]:
        """
        Determines automated policy enforcement action and logs security decision.
        """
        risk_score = self.calculate_contextual_risk(
            threat_probability=threat_probability,
            resource_sensitivity=resource_sensitivity,
            user_role=user_role,
            device_type=device_type,
            historical_risk=historical_risk,
            has_consent=has_consent
        )

        # Policy Thresholds & Decision Matrix (High-Severity Threat Mitigation takes precedence)
        if risk_score >= 0.80 or threat_probability >= 0.85:
            action = "QUARANTINE_DEVICE" if user_role == "iomt_device" else "BLOCK_ACTOR"
            decision = "DENY"
            reason = f"High severity threat detected (Risk: {risk_score:.3f}, ThreatProb: {threat_probability:.3f}). Active mitigation triggered."
        elif not has_consent and resource_sensitivity > 0.6:
            action = "REQUIRE_CONSENT"
            decision = "CHALLENGE"
            reason = "Sensitive PHI requested without verified active patient consent on ledger."
        elif risk_score >= 0.55 or threat_probability >= 0.50:
            action = "STEP_UP_AUTHENTICATION"
            decision = "CHALLENGE"
            reason = f"Moderate risk profile ({risk_score:.3f}). Elevating authentication to MFA and enhanced audit."
        elif risk_score >= 0.35:
            action = "LIMIT_RATE"
            decision = "ALLOW_CONSTRAINED"
            reason = f"Anomalous access frequency detected ({risk_score:.3f}). Constraining API rate limit."
        else:
            action = "ALLOW"
            decision = "PERMIT"
            reason = "Normal telemetry profile within zero-trust behavioral baseline."

        record = {
            "timestamp": time.time(),
            "actor_id": actor_id,
            "role": user_role,
            "device_id": device_id,
            "device_type": device_type,
            "resource_type": resource_type,
            "operation": operation,
            "threat_probability": float(round(threat_probability, 4)),
            "risk_score": float(round(risk_score, 4)),
            "decision": decision,
            "action": action,
            "reason": reason
        }

        # Append to audit log
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record
