#!/usr/bin/env python3
"""
scripts/generate_synthetic_ehr.py
Generates >100,000 valid, statistically informed, fully synthetic HL7 FHIR R4
patient bundles adhering to Phase 3 requirements.
"""

import os
import sys
import json
import uuid
import random
import argparse
import numpy as np
from datetime import datetime, date, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
SAMPLES_DIR = os.path.join(SYNTHETIC_DIR, "samples")
os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Synthetic pool data for realism without real patient PHI
FIRST_NAMES_FEMALE = [
    "Emma", "Olivia", "Sophia", "Ava", "Isabella", "Mia", "Amelia", "Harper", "Evelyn", "Abigail",
    "Emily", "Elizabeth", "Mila", "Ella", "Avery", "Sofia", "Camila", "Aria", "Scarlett", "Victoria",
    "Madison", "Luna", "Grace", "Chloe", "Penelope", "Layla", "Riley", "Zoey", "Nora", "Lily",
    "Eleanor", "Hannah", "Lillian", "Addison", "Aubrey", "Ellie", "Stella", "Natalie", "Zoe", "Leah",
    "Hazel", "Violet", "Aurora", "Savannah", "Audrey", "Brooklyn", "Bella", "Claire", "Skylar", "Lucy"
]

FIRST_NAMES_MALE = [
    "Liam", "Noah", "Oliver", "William", "James", "Benjamin", "Lucas", "Henry", "Alexander", "Mason",
    "Michael", "Ethan", "Daniel", "Jacob", "Logan", "Jackson", "Levi", "Sebastian", "Mateo", "Jack",
    "Owen", "Theodore", "Aiden", "Samuel", "Joseph", "John", "David", "Wyatt", "Matthew", "Luke",
    "Asher", "Carter", "Julian", "Grayson", "Leo", "Jayden", "Gabriel", "Isaac", "Lincoln", "Anthony",
    "Hudson", "Dylan", "Ezra", "Thomas", "Charles", "Christopher", "Jaxon", "Maverick", "Josiah", "Isaiah"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts"
]

CITIES = [
    ("Boston", "MA", "02115"), ("Cambridge", "MA", "02138"), ("Worcester", "MA", "01608"),
    ("Springfield", "MA", "01103"), ("Lowell", "MA", "01852"), ("Providence", "RI", "02903"),
    ("Hartford", "CT", "06103"), ("New Haven", "CT", "06510"), ("Portland", "ME", "04101")
]

ORGANIZATIONS = [
    {"name": "Massachusetts General Medical Center", "id": "org-hosp-a", "type": "prov"},
    {"name": "Brigham Specialty Health", "id": "org-hosp-b", "type": "prov"},
    {"name": "Metro Clinical Reference Laboratory", "id": "org-lab-1", "type": "lab"},
    {"name": "New England Academic Research Center", "id": "org-res-1", "type": "edu"}
]

PRACTITIONERS = [
    {"name": "Dr. Sarah Lin, MD", "id": "pract-101", "role": "Primary Care Physician"},
    {"name": "Dr. Marcus Vance, MD", "id": "pract-102", "role": "Cardiologist"},
    {"name": "Dr. Elena Rostova, MD", "id": "pract-103", "role": "Endocrinologist"},
    {"name": "Dr. David Kim, DO", "id": "pract-104", "role": "Emergency Physician"}
]

CONDITIONS_CATALOG = [
    {"code": "44054006", "display": "Type 2 diabetes mellitus", "system": "http://snomed.info/sct"},
    {"code": "38341003", "display": "Essential hypertension", "system": "http://snomed.info/sct"},
    {"code": "55822004", "display": "Hyperlipidemia", "system": "http://snomed.info/sct"},
    {"code": "195967001", "display": "Asthma", "system": "http://snomed.info/sct"},
    {"code": "232353008", "display": "Allergic rhinitis", "system": "http://snomed.info/sct"},
    {"code": "84757009", "display": "Epilepsy", "system": "http://snomed.info/sct"},
    {"code": "73211009", "display": "Diabetes mellitus", "system": "http://snomed.info/sct"},
    {"code": "69896004", "display": "Rheumatoid arthritis", "system": "http://snomed.info/sct"},
    {"code": "15777000", "display": "Prediabetes", "system": "http://snomed.info/sct"},
    {"code": "162864005", "display": "Body mass index 30+ - obesity", "system": "http://snomed.info/sct"}
]

MEDICATIONS_CATALOG = [
    {"code": "860975", "display": "Metformin hydrochloride 500 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "314076", "display": "Lisinopril 10 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "617314", "display": "Atorvastatin 20 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "745679", "display": "Albuterol 90 MCG/ACTUAT Inhalation Aerosol", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "310965", "display": "Levothyroxine Sodium 0.05 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "197361", "display": "Amlodipine 5 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "197806", "display": "Hydrochlorothiazide 25 MG Oral Tablet", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"},
    {"code": "855332", "display": "Omeprazole 20 MG Delayed Release Oral Capsule", "system": "http://www.nlm.nih.gov/research/umls/rxnorm"}
]

PROCEDURES_CATALOG = [
    {"code": "73761001", "display": "Colonoscopy", "system": "http://snomed.info/sct"},
    {"code": "71651007", "display": "Mammography", "system": "http://snomed.info/sct"},
    {"code": "268400002", "display": "12 lead electrocardiogram", "system": "http://snomed.info/sct"},
    {"code": "430193006", "display": "Medication reconciliation", "system": "http://snomed.info/sct"},
    {"code": "171207006", "display": "Depression screening", "system": "http://snomed.info/sct"}
]

ALLERGIES_CATALOG = [
    {"code": "91936005", "display": "Allergy to penicillin", "criticality": "high"},
    {"code": "300916003", "display": "Allergy to peanuts", "criticality": "high"},
    {"code": "294805000", "display": "Allergy to latex", "criticality": "low"},
    {"code": "91931000", "display": "Allergy to sulfonamide", "criticality": "medium"}
]

class SyntheticFHIRGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.ref_dist_path = os.path.join(METADATA_DIR, "clinical_reference_distributions.json")
        self.load_reference_distributions()

    def load_reference_distributions(self):
        if os.path.exists(self.ref_dist_path):
            with open(self.ref_dist_path) as f:
                self.ref_dist = json.load(f)
        else:
            self.ref_dist = {
                "gender_dist": {"female": 0.50, "male": 0.50},
                "birth_year_stats": {"min": 1930, "max": 2022, "mean": 1980, "std": 22}
            }

    def generate_patient_bundle(self, patient_idx: int) -> dict:
        rng = self.rng
        np_rng = self.np_rng

        patient_uuid = f"synth-pt-{patient_idx:07d}"
        gender = "female" if rng.random() < self.ref_dist["gender_dist"].get("female", 0.5) else "male"
        first_name = rng.choice(FIRST_NAMES_FEMALE if gender == "female" else FIRST_NAMES_MALE)
        last_name = rng.choice(LAST_NAMES)
        city, state, postal = rng.choice(CITIES)

        # Birth date distribution
        b_year = int(np.clip(np_rng.normal(1982, 18), 1930, 2022))
        b_month = rng.randint(1, 12)
        b_day = rng.randint(1, 28)
        birth_date = date(b_year, b_month, b_day)
        birth_date_iso = birth_date.isoformat()

        # Organization & Practitioner
        org_meta = rng.choice(ORGANIZATIONS)
        pract_meta = rng.choice(PRACTITIONERS)

        org_resource = {
            "resourceType": "Organization",
            "id": f"org-{uuid.uuid4().hex[:8]}",
            "name": org_meta["name"],
            "type": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/organization-type", "code": org_meta["type"]}]}]
        }

        pract_resource = {
            "resourceType": "Practitioner",
            "id": f"pract-{uuid.uuid4().hex[:8]}",
            "name": [{"family": pract_meta["name"].split()[-1], "given": [pract_meta["name"].split()[1].replace(',', '')]}],
            "identifier": [{"system": "http://hl7.org/fhir/sid/us-npi", "value": f"1{patient_idx % 900000000 + 100000000}"}]
        }

        # Patient Resource
        patient_resource = {
            "resourceType": "Patient",
            "id": patient_uuid,
            "identifier": [
                {"system": "https://github.com/crypto-agile-ehr/synthetic-id", "value": f"SYNTH-{patient_idx:07d}"},
                {"system": "http://hl7.org/fhir/sid/us-ssn", "value": f"999-{patient_idx % 90 + 10:02d}-{patient_idx % 9000 + 1000:04d}"}
            ],
            "active": True,
            "name": [{"use": "official", "family": last_name, "given": [first_name]}],
            "gender": gender,
            "birthDate": birth_date_iso,
            "address": [{"line": [f"{rng.randint(10, 999)} Synthetic Health Way"], "city": city, "state": state, "postalCode": postal, "country": "US"}],
            "managingOrganization": {"reference": f"Organization/{org_resource['id']}"}
        }

        # Timestamps for encounters (strictly after birthdate, strictly before 2026-10-01)
        ref_date = date(2026, 9, 15)
        start_enc_year = min(max(b_year, 2020), 2026)
        enc_year = rng.randint(start_enc_year, 2026)
        enc_month = rng.randint(1, 9 if enc_year == 2026 else 12)
        enc_day = rng.randint(1, 28)
        enc_start = datetime(enc_year, enc_month, enc_day, rng.randint(8, 16), rng.randint(0, 59), tzinfo=timezone.utc)
        enc_end = enc_start + timedelta(hours=rng.randint(1, 4), minutes=rng.randint(0, 45))

        enc_id = f"enc-{uuid.uuid4().hex[:8]}"
        encounter_resource = {
            "resourceType": "Encounter",
            "id": enc_id,
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "ambulatory"},
            "subject": {"reference": f"Patient/{patient_uuid}"},
            "participant": [{"individual": {"reference": f"Practitioner/{pract_resource['id']}"}}],
            "period": {"start": enc_start.isoformat(), "end": enc_end.isoformat()},
            "serviceProvider": {"reference": f"Organization/{org_resource['id']}"}
        }

        # Condition Resources (1 to 3 conditions)
        num_conds = rng.randint(1, 3)
        chosen_conds = rng.sample(CONDITIONS_CATALOG, num_conds)
        condition_resources = []
        for c in chosen_conds:
            c_onset_year = rng.randint(b_year, enc_year)
            c_res = {
                "resourceType": "Condition",
                "id": f"cond-{uuid.uuid4().hex[:8]}",
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis"}]}],
                "code": {"coding": [c]},
                "subject": {"reference": f"Patient/{patient_uuid}"},
                "encounter": {"reference": f"Encounter/{enc_id}"},
                "onsetDateTime": f"{c_onset_year}-0{rng.randint(1,9)}-15T09:00:00Z"
            }
            condition_resources.append(c_res)

        # Observations (Vitals & Labs)
        observation_resources = []
        
        # 1. Systolic Blood Pressure (LOINC 8480-6)
        sbp = float(np.clip(np_rng.normal(122, 14), 70, 210))
        # 2. Diastolic Blood Pressure (LOINC 8462-4)
        dbp = float(np.clip(np_rng.normal(78, 10), 45, 125))
        # 3. Heart Rate (LOINC 8867-4)
        hr = float(np.clip(np_rng.normal(74, 11), 45, 160))
        # 4. Blood Glucose (LOINC 2339-0)
        glucose = float(np.clip(np_rng.normal(104, 25), 60, 380))
        # 5. Hemoglobin (LOINC 718-7)
        hemoglobin = float(np.clip(np_rng.normal(14.2 if gender == "male" else 13.1, 1.3), 8.0, 18.5))
        # 6. Body Weight (LOINC 29463-7)
        weight = float(np.clip(np_rng.normal(76, 16), 40, 160))

        vitals = [
            ("8480-6", "Systolic blood pressure", round(sbp, 1), "mm[Hg]"),
            ("8462-4", "Diastolic blood pressure", round(dbp, 1), "mm[Hg]"),
            ("8867-4", "Heart rate", round(hr, 1), "/min"),
            ("2339-0", "Glucose [Mass/volume] in Blood", round(glucose, 1), "mg/dL"),
            ("718-7", "Hemoglobin [Mass/volume] in Blood", round(hemoglobin, 1), "g/dL"),
            ("29463-7", "Body weight", round(weight, 1), "kg")
        ]

        obs_refs = []
        for l_code, l_disp, l_val, l_unit in vitals:
            obs_id = f"obs-{uuid.uuid4().hex[:8]}"
            obs_refs.append({"reference": f"Observation/{obs_id}"})
            obs_res = {
                "resourceType": "Observation",
                "id": obs_id,
                "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
                "code": {"coding": [{"system": "http://loinc.org", "code": l_code, "display": l_disp}]},
                "subject": {"reference": f"Patient/{patient_uuid}"},
                "encounter": {"reference": f"Encounter/{enc_id}"},
                "effectiveDateTime": enc_start.isoformat(),
                "valueQuantity": {"value": l_val, "unit": l_unit, "system": "http://unitsofmeasure.org", "code": l_unit}
            }
            observation_resources.append(obs_res)

        # MedicationRequests (1 to 2)
        med_resources = []
        for m in rng.sample(MEDICATIONS_CATALOG, rng.randint(1, 2)):
            med_id = f"med-{uuid.uuid4().hex[:8]}"
            med_res = {
                "resourceType": "MedicationRequest",
                "id": med_id,
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"coding": [m]},
                "subject": {"reference": f"Patient/{patient_uuid}"},
                "encounter": {"reference": f"Encounter/{enc_id}"},
                "authoredOn": enc_start.isoformat(),
                "requester": {"reference": f"Practitioner/{pract_resource['id']}"}
            }
            med_resources.append(med_res)

        # Procedure (1)
        proc_catalog = rng.choice(PROCEDURES_CATALOG)
        proc_resource = {
            "resourceType": "Procedure",
            "id": f"proc-{uuid.uuid4().hex[:8]}",
            "status": "completed",
            "code": {"coding": [proc_catalog]},
            "subject": {"reference": f"Patient/{patient_uuid}"},
            "encounter": {"reference": f"Encounter/{enc_id}"},
            "performedDateTime": enc_start.isoformat()
        }

        # DiagnosticReport (1)
        diag_resource = {
            "resourceType": "DiagnosticReport",
            "id": f"diag-{uuid.uuid4().hex[:8]}",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074", "code": "LAB"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "58410-2", "display": "Complete blood count panel"}]},
            "subject": {"reference": f"Patient/{patient_uuid}"},
            "encounter": {"reference": f"Encounter/{enc_id}"},
            "effectiveDateTime": enc_start.isoformat(),
            "issued": enc_end.isoformat(),
            "performer": [{"reference": f"Organization/{org_resource['id']}"}],
            "result": obs_refs[:3]
        }

        # AllergyIntolerance (1)
        allergy_meta = rng.choice(ALLERGIES_CATALOG)
        allergy_resource = {
            "resourceType": "AllergyIntolerance",
            "id": f"allrg-{uuid.uuid4().hex[:8]}",
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification", "code": "confirmed"}]},
            "criticality": allergy_meta["criticality"],
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": allergy_meta["code"], "display": allergy_meta["display"]}]},
            "patient": {"reference": f"Patient/{patient_uuid}"}
        }

        # Build FHIR Bundle
        all_resources = [
            org_resource, pract_resource, patient_resource, encounter_resource,
            *condition_resources, *observation_resources, *med_resources,
            proc_resource, diag_resource, allergy_resource
        ]

        bundle = {
            "resourceType": "Bundle",
            "id": f"bundle-synth-{patient_idx:07d}",
            "meta": {
                "lastUpdated": datetime.now(timezone.utc).isoformat(),
                "profile": ["http://hl7.org/fhir/StructureDefinition/Bundle"]
            },
            "type": "transaction",
            "entry": [{"resource": res, "fullUrl": f"urn:uuid:{res['id']}"} for res in all_resources]
        }

        return bundle

def generate_dataset(total_count: int, shard_size: int = 10000, seed: int = 42):
    print("=" * 70)
    print(f"PHASE 3: GENERATING {total_count:,} SYNTHETIC FHIR R4 PATIENT BUNDLES")
    print(f"Random Master Seed: {seed}")
    print("=" * 70)

    generator = SyntheticFHIRGenerator(seed=seed)
    
    total_resources_generated = 0
    resource_type_breakdown = {}
    shards_written = []

    shard_idx = 0
    current_shard_records = []
    
    start_time = datetime.now()

    for idx in range(1, total_count + 1):
        bundle = generator.generate_patient_bundle(idx)
        current_shard_records.append(bundle)

        # Track resource counts
        for entry in bundle["entry"]:
            rt = entry["resource"]["resourceType"]
            resource_type_breakdown[rt] = resource_type_breakdown.get(rt, 0) + 1
            total_resources_generated += 1

        # Save first 50 as individual JSON files for REST API testing
        if idx <= 50:
            sample_path = os.path.join(SAMPLES_DIR, f"patient_bundle_{idx:05d}.json")
            with open(sample_path, "w") as fp:
                json.dump(bundle, fp, indent=2)

        # Write shard when reaching shard_size or end
        if len(current_shard_records) == shard_size or idx == total_count:
            shard_idx += 1
            shard_filename = f"fhir_patients_part_{shard_idx:03d}.jsonl"
            shard_path = os.path.join(SYNTHETIC_DIR, shard_filename)
            
            with open(shard_path, "w") as fp:
                for b in current_shard_records:
                    fp.write(json.dumps(b) + "\n")
                    
            shards_written.append({
                "shard": shard_filename,
                "path": shard_path,
                "records": len(current_shard_records),
                "size_bytes": os.path.getsize(shard_path)
            })
            
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = idx / elapsed if elapsed > 0 else 0
            print(f"  Shard {shard_idx:02d} written: {shard_filename} ({len(current_shard_records):,} bundles, {idx:,}/{total_count:,} total) | Rate: {rate:,.0f} bundles/s", end="\r")
            current_shard_records = []

    elapsed_total = (datetime.now() - start_time).total_seconds()
    print(f"\n[COMPLETE] Generated {total_count:,} patient bundles in {elapsed_total:.2f}s ({total_count/elapsed_total:,.1f} bundles/s).")
    print(f"Total FHIR Resources Generated: {total_resources_generated:,}")
    print("Resource Breakdown:")
    for rt, cnt in sorted(resource_type_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rt:20s}: {cnt:,} ({cnt/total_count:.1f} per patient)")

    # Save generation summary
    summary = {
        "dataset_name": "Synthetic-FHIR-R4-PostQuantum",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "master_seed": seed,
        "total_patient_bundles": total_count,
        "total_fhir_resources": total_resources_generated,
        "resource_breakdown": resource_type_breakdown,
        "shards": shards_written,
        "samples_directory": SAMPLES_DIR,
        "samples_count": min(total_count, 50),
        "generator_version": "1.0-research",
        "compliance": "HL7 FHIR Release 4 (R4)"
    }

    summary_path = os.path.join(SYNTHETIC_DIR, "generation_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\n[MANIFEST] Saved generation summary to {summary_path}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FHIR R4 dataset")
    parser.add_argument("--count", type=int, default=100000, help="Number of synthetic patient bundles (default: 100000)")
    parser.add_argument("--shard-size", type=int, default=10000, help="Records per shard file (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic master seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(total_count=args.count, shard_size=args.shard_size, seed=args.seed)

if __name__ == "__main__":
    main()
