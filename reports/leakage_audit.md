# Data Leakage Audit Report
**Standard:** Zero-Tolerance Leakage Protocol (Phase 3)
**Verification:** Features excluded prior to any feature engineering, splitting, or training.

## Excluded Leaky Features

| Dataset | Feature | Reason | Leakage Risk |
| :--- | :--- | :--- | :--- |
| Edge-IIoTset | `frame.time` | Temporal index / timestamp | Causes temporal lookahead and spurious temporal overfitting. |
| Edge-IIoTset | `ip.src_host` | Source IP address | Identity leak; model memorizes attacker IP instead of behavior. |
| Edge-IIoTset | `ip.dst_host` | Destination IP address | Identity leak; memorizes victim network infrastructure. |
| Edge-IIoTset | `arp.dst.proto_ipv4` | ARP destination IP | Network hardware identity leak. |
| Edge-IIoTset | `arp.src.proto_ipv4` | ARP source IP | Network hardware identity leak. |
| Edge-IIoTset | `Attack_label` | Target ground-truth label | Direct target proxy. |
| Edge-IIoTset | `Attack_type` | Target ground-truth attack family | Direct target proxy. |
| Synthetic-FHIR | `timestamp` | Epoch timestamp | Allows tree base-learners to overfit to attack capture windows rather than attack semantics. |
| Synthetic-FHIR | `source_dataset` | Dataset origin indicator | Identifies provenance trivially. |
| Synthetic-FHIR | `organization` | Hospital / Organization ID | Induces site-specific bias rather than domain generalization. |
| Synthetic-FHIR | `actor_id_hash` | Actor User Identifier Hash | Identifies specific attacking user account entity. |
| Synthetic-FHIR | `device_id_hash` | Device Identifier Hash | Memorizes device serial/MAC rather than telemetry dynamics. |
| Synthetic-FHIR | `patient_id_hash` | Patient Identifier Hash | Identifies specific targeted clinical record. |
| Synthetic-FHIR | `resource_id_hash` | Resource Identifier Hash | Identifies targeted resource entity directly. |
| Synthetic-FHIR | `attack_category` | Target attack category | Direct target proxy. |
| Synthetic-FHIR | `binary_label` | Target binary label | Direct target proxy. |
| CICIoT2023 | `label` | CICIoT ground-truth label | Direct target proxy. |

## Protocol Verification
- **No Lookahead:** All rolling and rate features are causally valid.
- **No Global Scaling:** Scalers and encoders are strictly fitted on the training split only.
- **Identity Stripping:** All IP addresses, hashes, entity IDs, and capture timestamps are purged.
