"""Canonical Schema and Dataset Mapping for HAB-IDS."""

from typing import Dict, List, Set

# Attack family taxonomy for CICIoT2023 (mapping from 34 granular types to 8 core families)
CICIOT_ATTACK_FAMILIES: Dict[str, str] = {
    "BenignTraffic": "Benign",
    # DDoS
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Mirai
    "Mirai-greeth_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Mirai-greip_flood": "Mirai",
    # Reconnaissance
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-PingSweep": "Recon",
    "VulnerabilityScan": "Recon",
    # Spoofing
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    # Web Attacks
    "SqlInjection": "Web",
    "CommandInjection": "Web",
    "BrowserHijacking": "Web",
    "XSS": "Web",
    "Uploading_Attack": "Web",
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
    # Malware
    "Backdoor_Malware": "Malware",
}

# Attack family taxonomy for Edge-IIoTset (15 granular classes to 7 core families)
EDGE_IIOT_ATTACK_FAMILIES: Dict[str, str] = {
    "Normal": "Benign",
    "DDoS_UDP": "DDoS",
    "DDoS_ICMP": "DDoS",
    "DDoS_HTTP": "DDoS",
    "DDoS_TCP": "DDoS",
    "Ransomware": "Malware",
    "Backdoor": "Malware",
    "SQL_injection": "Web",
    "Uploading": "Web",
    "XSS": "Web",
    "Vulnerability_scanner": "Recon",
    "Port_Scanning": "Recon",
    "Fingerprinting": "Recon",
    "Password": "BruteForce",
    "MITM": "Spoofing",
}

# Attack family taxonomy for FHIR Security Events
FHIR_ATTACK_FAMILIES: Dict[str, str] = {
    "normal_access": "Benign",
    "repeated_authentication_failure": "BruteForce",
    "credential_compromise": "BruteForce",
    "privilege_escalation": "PrivilegeEscalation",
    "unauthorized_access": "UnauthorizedAccess",
    "resource_enumeration": "Recon",
    "abnormal_access_frequency": "Anomaly",
    "abnormal_device_behavior": "Anomaly",
    "insider_like_behavior": "Anomaly",
    "bulk_record_access": "Exfiltration",
    "data_exfiltration_pattern": "Exfiltration",
    "tampering_attempt": "IntegrityAttack",
    "FHIR_API_abuse": "Web",
    "suspicious_location": "Anomaly",
    "unusual_time_access": "Anomaly",
}

# Cross-Domain Harmonized Features (Features legitimately computable across network/telemetry domains)
COMMON_CROSS_DOMAIN_FEATURES: List[str] = [
    "flow_duration",
    "packet_count",
    "byte_count",
    "packet_rate",
    "byte_rate",
    "dst_port",
    "protocol",
]

# Leakage exclusion list: features that contain identity, timestamps, or target encodings
STRICT_LEAKAGE_EXCLUSIONS: Set[str] = {
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "timestamp",
    "actor_id_hash",
    "patient_id_hash",
    "resource_id_hash",
    "device_id_hash",
    "source_dataset",
    "organization",
    "Attack_label",
    "Attack_type",
    "binary_label",
    "attack_category",
    "label",
}


def get_ciciot_features(include_label: bool = False) -> List[str]:
    """Return all valid feature columns for CICIoT2023 without leakage."""
    features = [
        "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
        "Srate", "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
        "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
        "ack_count", "syn_count", "fin_count", "urg_count", "rst_count", "HTTP",
        "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
        "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std",
        "Tot size", "IAT", "Number", "Magnitue", "Radius", "Covariance",
        "Variance", "Weight"
    ]
    if include_label:
        features.append("label")
    return features


def get_edge_iiot_features(include_labels: bool = False) -> List[str]:
    """Return valid feature columns for Edge-IIoTset excluding leaky ID/time columns."""
    features = [
        "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le",
        "icmp.transmit_timestamp", "icmp.unused", "http.content_length",
        "http.response", "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.checksum",
        "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
        "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack",
        "tcp.len", "tcp.seq", "tcp.srcport", "udp.port", "udp.stream",
        "udp.time_delta", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
        "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
        "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
        "mqtt.hdrflags", "mqtt.len", "mqtt.msgtype", "mqtt.proto_len",
        "mqtt.topic_len", "mqtt.ver", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id"
    ]
    if include_labels:
        features.extend(["Attack_label", "Attack_type"])
    return features


def get_fhir_features(include_labels: bool = False) -> List[str]:
    """Return valid feature columns for Synthetic FHIR without leakage."""
    features = [
        "user_role", "device_type", "resource_type", "operation",
        "resource_sensitivity", "auth_status", "failed_auth_count",
        "request_frequency", "burst_score", "historical_risk", "dst_port",
        "protocol", "flow_duration", "packet_count", "byte_count",
        "packet_rate", "byte_rate"
    ]
    if include_labels:
        features.extend(["binary_label", "attack_category"])
    return features
