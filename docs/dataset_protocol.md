# Dataset Protocol: Acquisition, Synthesis, Provenance, and Separation

## 1. Scope and Separation Mandate
This research project employs three strictly decoupled data streams. Under no circumstances may data from different streams be merged, intermixed, or conflated:
- **Stream A (Reference Clinical EHR Schema)**: MIMIC-IV-on-FHIR (PhysioNet credentialed research data schema) used exclusively to derive realistic resource structures, terminology bindings (LOINC, SNOMED, RxNorm), and clinical constraints.
- **Stream B (Synthetic Clinical EHR Data)**: Synthea/SyntheticMass methodology generating >100,000 (targeting 250,000+) fully synthetic, valid HL7 FHIR R4 records. Seeded, deterministic, containing zero real patient identifiers.
- **Stream C (Cybersecurity Telemetry Data)**: Edge-IIoTset (comprehensive cybersecurity dataset for IoT and network environments) used exclusively for network and API threat detection, intrusion classification, and anomaly modeling.

```
       +-------------------------------------------------------------------------+
       |                           DATA STREAM PARTITION                         |
       +-------------------------------------------------------------------------+
       |                                                                         |
       |  +---------------------------+       +-------------------------------+  |
       |  |  Stream A & B: Clinical   |       |  Stream C: Telemetry          |  |
       |  |  FHIR EHR Records         |       |  Edge-IIoTset Network Flow    |  |
       |  |  - Patient                |       |  - Packet size statistics     |  |
       |  |  - Encounter              |       |  - Inter-arrival times        |  |
       |  |  - Condition              |       |  - Protocol flags (TCP/UDP)   |  |
       |  |  - Observation            |       |  - API error frequencies      |  |
       |  |  - MedicationRequest      |       |  - Flow duration              |  |
       |  |  - DiagnosticReport       |       |  - Attack labels              |  |
       |  +-------------+-------------+       +---------------+---------------+  |
       |                |                                     |                  |
       |                v                                     v                  |
       |  Used In: Crypto, Storage,           Used In: TCN-Transformer,         |
       |  Blockchain, Minimization            Anomaly Detection, SHAP            |
       |                                                                         |
       +-------------------------------------------------------------------------+
```

---

## 2. Stream A: Reference Clinical EHR Protocol (MIMIC-IV-on-FHIR)
- **Source**: PhysioNet / Johnson et al.
- **License**: PhysioNet Credentialed Health Data License 1.5.0.
- **Governance Requirement**: Direct automatic download requires signed DUA (Data Use Agreement) and CITI training. Therefore, reference structural schemas, profile definitions, and distribution parameters are extracted from the public MIMIC-IV-on-FHIR specification repository and synthetic demo releases.
- **Purpose**: Establishes reference vocabulary systems (ICD-10-CM, LOINC, SNOMED-CT, RxNorm) and realistic resource relationship cardinalities.

---

## 3. Stream B: Large-Scale Synthetic EHR Generation Protocol (>100,000 Records)
### 3.1 Synthesis Engine Requirements
- **Target Record Count**: $N \ge 100,000$ synthetic patient bundles (scalable to 250,000+).
- **FHIR Version**: HL7 FHIR Release 4 (R4).
- **Seed Management**: Deterministic generation using PRNG with master seed `42` (logged in `data/metadata/dataset_manifest.json`).
- **Resource Composition**: Every synthetic patient bundle must preserve realistic clinical graph relationships:
  - `Patient` (Demographics, synthetic identifier, birthdate, gender)
  - `Encounter` (Ambulatory, inpatient, emergency, with start/end timestamps)
  - `Condition` (Clinical status, verification status, SNOMED-CT / ICD-10 code, onset date)
  - `Observation` (Vital signs, lab panels: systolic/diastolic BP, heart rate, blood glucose, hemoglobin, potassium, sodium, with LOINC codes, values, and units)
  - `MedicationRequest` (Status, intent, authoredOn, RxNorm codes)
  - `Procedure` (Status, SNOMED-CT procedure code, performed date)
  - `DiagnosticReport` (Status, code, category, issued date, linked observations)
  - `AllergyIntolerance` (Clinical status, criticality, substance code)
  - `Organization` & `Practitioner` (Fictitious institutions, NPI references)

### 3.2 Clinical Plausibility & Temporal Constraints
1. **Birthdate $\le$ Encounter Start $\le$ Encounter End $\le$ Current Timestamp**.
2. **Condition Onset $\ge$ Birthdate**.
3. **Observations and MedicationRequests must occur within valid encounter boundaries**.
4. **Physiological Range Boundaries**:
   - Systolic BP: $60 - 240$ mmHg
   - Diastolic BP: $40 - 140$ mmHg
   - Heart Rate: $40 - 200$ bpm
   - Blood Glucose: $50 - 500$ mg/dL
   - Hemoglobin: $6.0 - 20.0$ g/dL

---

## 4. Stream C: Cybersecurity Telemetry Protocol (Edge-IIoTset)
- **Source**: Ferrag et al., IEEE Transactions on Information Forensics and Security / IEEE Dataport.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Contents**: High-dimensional network flow data containing normal traffic and 14 distinct cyber attack types:
  - DoS/DDoS (HTTP, UDP, TCP SYN)
  - Injection (SQLi, XSS, Command Injection)
  - Malware (Backdoor, Ransomware)
  - Reconnaissance (Port Scanning, OS Fingerprinting)
  - Man-in-the-Middle (MITM / ARP Spoofing)
- **Abstraction Layer**: Maps raw network flow telemetry into a normalized security event feature vector:
  - Traffic rate metrics (bytes/sec, packets/sec)
  - Packet size distribution (min, max, mean, std)
  - Protocol state flags and connection duration
  - API invocation frequency and error rate burst indices
  - Cryptographic anomaly indicators (corrupted tag count, replay nonce count)

---

## 5. Statistical Validation Protocol
Prior to downstream experiments, the synthetic dataset must pass rigorous distribution and schema validation:
1. **Schema Integrity**: 100% compliance with FHIR R4 schema models.
2. **Identifier Uniqueness**: 0 duplicate UUIDs across all generated resources.
3. **Temporal Consistency**: 0 temporal violations (e.g., encounter preceding birthdate).
4. **Distribution Testing**:
   - Numerical features evaluated using Kolmogorov-Smirnov (KS) test and Jensen-Shannon (JS) divergence.
   - Categorical features evaluated using Chi-Square ($\chi^2$) goodness-of-fit.
5. **Validation Artifact**: Comprehensive HTML validation report generated at `reports/dataset_validation_report.html`.
