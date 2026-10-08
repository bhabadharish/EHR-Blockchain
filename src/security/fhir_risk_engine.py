"""FHIR Contextual Risk Engine for HAB-IDS (Phase 22).

Independent clinical security assessment engine combining:
- ML threat probability
- Authentication status & failure severity
- FHIR operation risk profile (e.g., bulk export / mutation)
- Target clinical resource sensitivity (e.g., DiagnosticReport / MedicationRequest)
- Request rate and historical telemetry anomalies
Generates contextual risk score [0, 100], risk category (LOW, MEDIUM, HIGH, CRITICAL),
and audit reason codes.
Operates downstream and decoupled from raw ML model benchmark evaluation.
"""

from typing import Dict, Any, List


class FHIRRiskEngine:
    """Decoupled clinical risk evaluation engine."""

    OPERATION_WEIGHTS = {
        "read": 1.0,
        "vread": 1.0,
        "search": 1.5,
        "create": 2.0,
        "update": 2.5,
        "delete": 4.0,
        "batch": 3.5,
        "transaction": 4.0,
    }

    RESOURCE_SENSITIVITY_BASE = {
        "Observation": 0.6,
        "DiagnosticReport": 0.9,
        "MedicationRequest": 0.8,
        "Condition": 0.7,
        "Patient": 0.85,
        "Consent": 0.95,
        "Provenance": 0.5,
        "AuditEvent": 0.6,
        "Organization": 0.3,
        "Device": 0.4,
    }

    def evaluate_risk(
        self,
        ml_probability: float,
        auth_status: int,
        failed_auth_count: int,
        operation: str,
        resource_type: str,
        request_frequency: float,
        burst_score: float,
        historical_risk: float
    ) -> Dict[str, Any]:
        """Compute composite clinical risk score and actionable reason codes."""
        reason_codes: List[str] = []

        # 1. Base ML Threat Component (40% weight)
        ml_score = float(ml_probability * 40.0)
        if ml_probability > 0.75:
            reason_codes.append("RC_ML_CONFIDENT_ATTACK")
        elif ml_probability > 0.40:
            reason_codes.append("RC_ML_ELEVATED_RISK")

        # 2. Authentication Severity (20% weight)
        auth_score = 0.0
        if auth_status == 0:
            auth_score += 10.0
            reason_codes.append("RC_AUTH_FAILED")
        if failed_auth_count >= 5:
            auth_score += 10.0
            reason_codes.append("RC_BRUTE_FORCE_AUTHENTICATION")
        elif failed_auth_count > 0:
            auth_score += min(failed_auth_count * 2.0, 8.0)

        # 3. Clinical Operation & Resource Sensitivity (20% weight)
        op_weight = self.OPERATION_WEIGHTS.get(operation.lower(), 1.0)
        res_sens = self.RESOURCE_SENSITIVITY_BASE.get(resource_type, 0.5)
        op_res_score = min((op_weight / 4.0) * res_sens * 20.0, 20.0)
        if op_weight >= 3.5:
            reason_codes.append("RC_BULK_OR_TRANSACTION_OPERATION")
        if res_sens >= 0.85:
            reason_codes.append("RC_CRITICAL_PHI_TARGETED")

        # 4. Telemetry Behavioral Anomaly (20% weight)
        telemetry_score = 0.0
        if burst_score > 0.7:
            telemetry_score += 10.0
            reason_codes.append("RC_ABNORMAL_TRAFFIC_BURST")
        if request_frequency > 20.0:
            telemetry_score += 10.0
            reason_codes.append("RC_HIGH_FREQUENCY_ENUMERATION")

        total_risk_score = round(ml_score + auth_score + op_res_score + telemetry_score, 2)
        total_risk_score = min(max(total_risk_score, 0.0), 100.0)

        # Categorization
        if total_risk_score < 25.0:
            category = "LOW"
        elif total_risk_score < 55.0:
            category = "MEDIUM"
        elif total_risk_score < 80.0:
            category = "HIGH"
        else:
            category = "CRITICAL"

        return {
            "risk_score": total_risk_score,
            "risk_category": category,
            "reason_codes": reason_codes,
            "subscores": {
                "ml_score": round(ml_score, 2),
                "auth_score": round(auth_score, 2),
                "op_res_score": round(op_res_score, 2),
                "telemetry_score": round(telemetry_score, 2),
            }
        }
