"""
src/inference/validator.py
==========================
Event validation layer for real-time FHIR threat inference.
"""

from typing import Dict, Any, Tuple

REQUIRED_EVENT_KEYS = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "byte_rate",
    "dst_port",
    "protocol",
    "resource_sensitivity",
    "auth_status",
    "failed_auth_count",
    "request_frequency",
    "burst_score",
    "historical_risk",
    "user_role",
    "resource_type",
    "operation"
]

def validate_event_schema(event: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates incoming event schema before model ingestion.
    """
    for key in REQUIRED_EVENT_KEYS:
        if key not in event:
            return False, f"Missing required field: '{key}'"

    try:
        float(event["flow_duration"])
        float(event["packet_count"])
        float(event["byte_count"])
        float(event["packet_rate"])
        float(event["byte_rate"])
        int(event["dst_port"])
        int(event["protocol"])
        float(event["resource_sensitivity"])
        int(event["auth_status"])
        int(event["failed_auth_count"])
        float(event["request_frequency"])
        float(event["burst_score"])
        float(event["historical_risk"])
    except (ValueError, TypeError) as e:
        return False, f"Invalid numerical field format: {e}"

    if str(event["user_role"]).strip() == "":
        return False, "user_role cannot be empty"
    if str(event["resource_type"]).strip() == "":
        return False, "resource_type cannot be empty"
    if str(event["operation"]).strip() == "":
        return False, "operation cannot be empty"

    return True, "Valid"
