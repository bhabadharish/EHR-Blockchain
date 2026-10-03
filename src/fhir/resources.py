import json
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple

class FHIRResourceManager:
    """
    Standard FHIR R4 Resource Builder and Canonical Normalizer.
    Generates standard compliant JSON representations for healthcare telemetry.
    """

    @staticmethod
    def create_patient(patient_id: str, name: str, gender: str, birth_date: str) -> Dict[str, Any]:
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {"versionId": "1", "lastUpdated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "active": True,
            "name": [{"use": "official", "family": name.split()[-1], "given": [name.split()[0]]}],
            "gender": gender,
            "birthDate": birth_date
        }

    @staticmethod
    def create_observation(
        obs_id: str,
        patient_id: str,
        code_loinc: str,
        display: str,
        value: float,
        unit: str,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code_loinc, "display": display}]},
            "subject": {"reference": f"Patient/{patient_id}"},
            "device": {"reference": f"Device/{device_id}"} if device_id else None,
            "effectiveDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"}
        }

    @staticmethod
    def create_consent(
        consent_id: str,
        patient_id: str,
        grantee_role: str,
        allowed_resources: List[str],
        expires_days: int = 30
    ) -> Dict[str, Any]:
        return {
            "resourceType": "Consent",
            "id": consent_id,
            "status": "active",
            "scope": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/consentscope", "code": "patient-privacy"}]},
            "category": [{"coding": [{"system": "http://loinc.org", "code": "59284-0"}]}],
            "patient": {"reference": f"Patient/{patient_id}"},
            "dateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provision": {
                "type": "permit",
                "period": {
                    "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + expires_days * 86400))
                },
                "actor": [{"role": {"coding": [{"code": grantee_role}]}}],
                "action": [{"coding": [{"code": "access"}]}],
                "class": [{"code": r} for r in allowed_resources]
            }
        }

    @staticmethod
    def canonical_json_string(resource: Dict[str, Any]) -> str:
        """Produces deterministic canonical string for cryptographic hashing."""
        return json.dumps(resource, sort_keys=True, separators=(',', ':'))

class FHIRValidator:
    """
    Validates structural integrity and mandatory FHIR R4 schema elements.
    """
    REQUIRED_FIELDS = {
        "Patient": ["resourceType", "id", "name"],
        "Observation": ["resourceType", "id", "status", "code", "subject"],
        "Consent": ["resourceType", "id", "status", "patient", "provision"],
        "Device": ["resourceType", "id"],
        "AuditEvent": ["resourceType", "id", "type", "agent", "recorded"]
    }

    @classmethod
    def validate(cls, resource: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(resource, dict):
            return False, "Payload must be a JSON dictionary."
        
        rtype = resource.get("resourceType")
        if not rtype:
            return False, "Missing mandatory 'resourceType' field."

        req_fields = cls.REQUIRED_FIELDS.get(rtype, ["resourceType", "id"])
        missing = [f for f in req_fields if f not in resource or resource[f] is None]
        if missing:
            return False, f"Resource '{rtype}' missing mandatory fields: {missing}"

        return True, "Valid FHIR R4 structure."
