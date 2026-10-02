#!/usr/bin/env python3
"""Generate a reproducible 50,000-row synthetic 5G network-log dataset.

The generator uses only Python's standard library. It creates realistic text
for TF-IDF and Word2Vec experiments while preserving exact class, severity and
train/validation/test distributions. Research-only metadata must not be used
as model input unless it is part of a separately documented experiment.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path



TOTAL_LOGS = 50_000
DEFAULT_SEED = 7_002_026
FIELDNAMES = [
    "timestamp",
    "network_function",
    "severity",
    "log_text",
    "label",
    "anomaly_type",
    "scenario_id",
    "session_id",
    "template_id",
    "split",
]

CLASS_COUNTS = {"normal": 37_500, "abnormal": 12_500}
SPLIT_COUNTS = {"train": 35_000, "validation": 7_500, "test": 7_500}
SPLIT_LABEL_COUNTS = {
    ("train", "normal"): 26_250,
    ("train", "abnormal"): 8_750,
    ("validation", "normal"): 5_625,
    ("validation", "abnormal"): 1_875,
    ("test", "normal"): 5_625,
    ("test", "abnormal"): 1_875,
}

# Exactly 500 normal WARNING rows and 500 abnormal INFO rows form the 2% severity exceptions.
SEVERITY_QUOTAS = {
    ("train", "normal"): {"INFO": 18_900, "DEBUG": 4_200, "NOTICE": 2_800, "WARNING": 350},
    ("validation", "normal"): {"INFO": 4_050, "DEBUG": 900, "NOTICE": 600, "WARNING": 75},
    ("test", "normal"): {"INFO": 4_050, "DEBUG": 900, "NOTICE": 600, "WARNING": 75},
    ("train", "abnormal"): {"INFO": 350, "WARNING": 3_150, "ERROR": 4_550, "CRITICAL": 700},
    ("validation", "abnormal"): {"INFO": 75, "WARNING": 675, "ERROR": 975, "CRITICAL": 150},
    ("test", "abnormal"): {"INFO": 75, "WARNING": 675, "ERROR": 975, "CRITICAL": 150},
}


@dataclass(frozen=True)
class EventSpec:
    """Message patterns and contextual endings for one operational event."""

    templates: tuple[tuple[str, str], ...]
    details: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioPlan:
    """A related group of rows kept wholly inside one dataset split."""

    split: str
    label: str
    event_name: str
    size: int


# Four message patterns combined with four details give 16 templates per event.
# Detail 4 is reserved for part of the test set to assess unfamiliar wording.
EVENT_SPECS: dict[str, EventSpec] = {
    "ue_registration_success": EventSpec(
        templates=(
            ("AMF", "Registration request from UE {supi} through {gnb} passed identity and access checks"),
            ("AMF", "UE {supi} completed registration and received temporary identity {guti}"),
            ("UDM", "Subscription data for {supi} was retrieved for registration on slice {slice_id}"),
            ("AUSF", "Authentication response for {supi} was verified and registration may continue"),
        ),
        details=(
            "mobility state changed to registered",
            "serving cell {cell_id} reported stable radio conditions",
            "requested network slice {slice_id} remained available",
            "registration procedure finished in {latency_ms} milliseconds",
        ),
    ),
    "authentication_success": EventSpec(
        templates=(
            ("AUSF", "Authentication vector was generated successfully for subscriber {supi}"),
            ("UDM", "Subscriber credentials for {supi} matched the stored authentication profile"),
            ("AMF", "Security mode procedure completed for UE {supi} using integrity algorithm {algorithm}"),
            ("UE", "Authentication challenge from serving network was answered successfully for session {session_id}"),
        ),
        details=(
            "sequence counter remained within the accepted window",
            "integrity verification completed without a mismatch",
            "security context {security_context} became active",
            "authentication response arrived after {latency_ms} milliseconds",
        ),
    ),
    "pdu_session_success": EventSpec(
        templates=(
            ("SMF", "PDU session request for {supi} was accepted on data network {dnn}"),
            ("PCF", "Policy association {policy_id} was installed for session {session_id}"),
            ("UPF", "GTP-U tunnel with TEID {teid} became active for {session_id}"),
            ("SMF", "QoS flow {qfi} was created for slice {slice_id} and data network {dnn}"),
        ),
        details=(
            "allocated address {ue_ip} passed reachability checks",
            "user-plane forwarding rules were applied consistently",
            "requested QoS profile {qos_profile} was granted",
            "session establishment completed in {latency_ms} milliseconds",
        ),
    ),
    "handover_success": EventSpec(
        templates=(
            ("gNB", "Handover preparation for UE {supi} from {source_gnb} to {target_gnb} completed"),
            ("AMF", "Mobility context for {supi} was transferred to target node {target_gnb}"),
            ("UPF", "Downlink forwarding path was updated after handover of session {session_id}"),
            ("UE", "UE {supi} confirmed attachment to target cell {target_cell_id}"),
        ),
        details=(
            "source radio bearer was released after confirmation",
            "packet forwarding continued without interruption",
            "target signal level measured {signal_dbm} dBm",
            "handover execution finished in {latency_ms} milliseconds",
        ),
    ),
    "service_discovery_success": EventSpec(
        templates=(
            ("NRF", "Network function {nf_instance} registered its service profile and endpoint {service_ip}"),
            ("SCP", "Service request for {service_name} was routed to healthy instance {nf_instance}"),
            ("NRF", "Discovery query returned an available {target_nf} instance for requester {requester_nf}"),
            ("AMF", "AMF resolved service {service_name} through NRF endpoint {service_ip}"),
        ),
        details=(
            "service validity remains active for {validity_seconds} seconds",
            "endpoint priority {priority} satisfied routing policy",
            "registered capacity was reported as {capacity_pct} percent",
            "discovery response arrived in {latency_ms} milliseconds",
        ),
    ),
    "heartbeat_health": EventSpec(
        templates=(
            ("NRF", "Heartbeat response from network function {nf_instance} confirmed service availability"),
            ("SCP", "Health probe for endpoint {service_ip} completed successfully"),
            ("UPF", "User-plane heartbeat for node {nf_instance} returned before expiry"),
            ("SMF", "Session management service {nf_instance} renewed its status with NRF"),
        ),
        details=(
            "next health check is scheduled after {heartbeat_seconds} seconds",
            "consecutive successful probe count reached {success_count}",
            "service capacity remained at {capacity_pct} percent",
            "heartbeat round-trip time was {latency_ms} milliseconds",
        ),
    ),
    "policy_update_success": EventSpec(
        templates=(
            ("PCF", "Policy rule {policy_id} was updated for subscriber {supi}"),
            ("SMF", "Session {session_id} applied the latest policy decision from PCF"),
            ("NSSF", "Slice selection for {supi} returned allowed slice {slice_id}"),
            ("AMF", "Access restriction update for UE {supi} was acknowledged"),
        ),
        details=(
            "existing QoS flow {qfi} remained active",
            "policy version advanced to revision {policy_version}",
            "configured usage limit is {usage_limit_mb} megabytes",
            "policy acknowledgement arrived in {latency_ms} milliseconds",
        ),
    ),
    "qos_monitoring_normal": EventSpec(
        templates=(
            ("UPF", "QoS flow {qfi} carried traffic within the configured bandwidth profile"),
            ("SMF", "Session {session_id} reported stable throughput and delay measurements"),
            ("PCF", "QoS policy {policy_id} remained satisfied for slice {slice_id}"),
            ("gNB", "Radio bearer for UE {supi} maintained the requested service level"),
        ),
        details=(
            "measured throughput was {throughput_mbps} megabits per second",
            "packet loss remained below {low_loss_pct} percent",
            "average queue utilisation was {low_queue_pct} percent",
            "measurement interval closed after {monitor_seconds} seconds",
        ),
    ),
    "authentication_failure": EventSpec(
        templates=(
            ("AUSF", "Authentication response for subscriber {supi} failed integrity verification"),
            ("AMF", "Security mode procedure for UE {supi} was rejected after invalid authentication data"),
            ("UDM", "Authentication sequence for {supi} exceeded the permitted resynchronisation window"),
            ("AUSF", "Repeated authentication attempts from UE {supi} exhausted the retry allowance"),
        ),
        details=(
            "failure counter increased to {retry_count}",
            "security context {security_context} was not activated",
            "access was withheld pending credential validation",
            "authentication procedure exceeded {latency_ms} milliseconds",
        ),
    ),
    "registration_rejection": EventSpec(
        templates=(
            ("AMF", "Registration request from UE {supi} was rejected with cause {cause_code}"),
            ("UDM", "Subscriber {supi} was not authorised for the requested serving network"),
            ("NSSF", "No permitted network slice was found for registration of {supi}"),
            ("AMF", "Registration procedure for UE {supi} ended after repeated context validation failures"),
        ),
        details=(
            "temporary identity {guti} was released",
            "requested slice was {slice_id}",
            "registration retry count reached {retry_count}",
            "procedure remained unresolved for {latency_ms} milliseconds",
        ),
    ),
    "nrf_discovery_failure": EventSpec(
        templates=(
            ("NRF", "Discovery request for service {service_name} returned no available {target_nf} instance"),
            ("SCP", "Service routing to {target_nf} failed because registered endpoints were unreachable"),
            ("AMF", "AMF could not resolve service {service_name} through NRF endpoint {service_ip}"),
            ("NRF", "Network function profile for {nf_instance} expired before renewal was received"),
        ),
        details=(
            "discovery retry count reached {retry_count}",
            "last endpoint response carried status {http_status}",
            "service selection queue contained {queue_depth} pending requests",
            "discovery operation exceeded {latency_ms} milliseconds",
        ),
    ),
    "packet_loss": EventSpec(
        templates=(
            ("UPF", "User-plane packet loss for session {session_id} increased to {packet_loss_pct} percent"),
            ("gNB", "Radio bearer for UE {supi} dropped {packet_count} packets during the measurement window"),
            ("SMF", "Session {session_id} reported persistent forwarding loss on the selected UPF path"),
            ("UPF", "GTP-U tunnel {teid} discarded packets after repeated sequence gaps"),
        ),
        details=(
            "affected QoS flow was {qfi}",
            "loss persisted across {monitor_seconds} seconds",
            "retransmission count increased to {retry_count}",
            "measured throughput fell to {degraded_throughput_mbps} megabits per second",
        ),
    ),
    "resource_exhaustion": EventSpec(
        templates=(
            ("UPF", "Packet-processing queue reached {high_queue_pct} percent utilisation on node {nf_instance}"),
            ("AMF", "UE context capacity reached {capacity_pct} percent and new requests were delayed"),
            ("SMF", "Session table usage exceeded the operational threshold on instance {nf_instance}"),
            ("NRF", "Service registry worker pool exhausted available processing slots"),
        ),
        details=(
            "pending operation count reached {queue_depth}",
            "memory utilisation increased to {memory_pct} percent",
            "request processing delay reached {latency_ms} milliseconds",
            "capacity protection activated for {monitor_seconds} seconds",
        ),
    ),
    "excessive_latency": EventSpec(
        templates=(
            ("gNB", "Radio access delay for UE {supi} exceeded the configured latency objective"),
            ("UPF", "User-plane forwarding latency for session {session_id} reached {latency_ms} milliseconds"),
            ("SMF", "PDU session operation remained pending beyond the allowed response interval"),
            ("AMF", "Registration signalling for UE {supi} experienced repeated processing delays"),
        ),
        details=(
            "affected slice was {slice_id}",
            "queue depth increased to {queue_depth} messages",
            "observed value exceeded threshold {latency_threshold_ms} milliseconds",
            "delay persisted for {monitor_seconds} seconds",
        ),
    ),
    "handover_failure": EventSpec(
        templates=(
            ("gNB", "Handover of UE {supi} from {source_gnb} to {target_gnb} failed during preparation"),
            ("AMF", "Mobility context transfer for {supi} was rejected by target node {target_gnb}"),
            ("UPF", "Forwarding path update for session {session_id} failed after target handover"),
            ("UE", "UE {supi} did not confirm attachment to target cell {target_cell_id}"),
        ),
        details=(
            "source bearer remained active pending recovery",
            "handover cause was {cause_code}",
            "target signal measured {weak_signal_dbm} dBm",
            "mobility operation timed out after {latency_ms} milliseconds",
        ),
    ),
    "signalling_storm": EventSpec(
        templates=(
            ("AMF", "Registration request rate from {gnb} increased to {request_rate} messages per second"),
            ("SCP", "Service-control traffic exceeded the permitted rate for requester {requester_nf}"),
            ("NRF", "Discovery requests for {service_name} formed a rapidly growing processing queue"),
            ("AMF", "Repeated signalling from UE group {ue_group} triggered rate protection"),
        ),
        details=(
            "rate-limiter threshold was {rate_threshold} messages per second",
            "pending queue depth reached {queue_depth}",
            "request count remained elevated for {monitor_seconds} seconds",
            "control-plane utilisation increased to {capacity_pct} percent",
        ),
    ),
    "malformed_message": EventSpec(
        templates=(
            ("AMF", "NGAP message from {gnb} contained an invalid {invalid_field} field"),
            ("SMF", "Session request for {session_id} failed schema validation at field {invalid_field}"),
            ("UPF", "GTP control message for tunnel {teid} carried an unsupported information element"),
            ("SCP", "Service request from {requester_nf} could not be parsed as a valid SBI payload"),
        ),
        details=(
            "message was rejected with cause {cause_code}",
            "decoder stopped at byte offset {byte_offset}",
            "protocol version was reported as {protocol_version}",
            "validation recorded {validation_errors} structural errors",
        ),
    ),
    "slice_policy_violation": EventSpec(
        templates=(
            ("NSSF", "Subscriber {supi} requested disallowed network slice {slice_id}"),
            ("PCF", "Policy rule {policy_id} denied the requested QoS level for {session_id}"),
            ("SMF", "PDU session {session_id} violated the configured slice bandwidth limit"),
            ("AMF", "Access request from UE {supi} conflicted with slice restriction policy"),
        ),
        details=(
            "permitted slice was {allowed_slice_id}",
            "observed throughput reached {throughput_mbps} megabits per second",
            "policy version was {policy_version}",
            "violation remained active for {monitor_seconds} seconds",
        ),
    ),
}


# Counts intentionally cover varied normal procedures while balancing ten anomaly types.
EVENT_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "ue_registration_success": {"train": 4_375, "validation": 938, "test": 937},
    "authentication_success": {"train": 3_150, "validation": 675, "test": 675},
    "pdu_session_success": {"train": 4_375, "validation": 937, "test": 938},
    "handover_success": {"train": 3_150, "validation": 675, "test": 675},
    "service_discovery_success": {"train": 3_150, "validation": 675, "test": 675},
    "heartbeat_health": {"train": 3_150, "validation": 675, "test": 675},
    "policy_update_success": {"train": 2_450, "validation": 525, "test": 525},
    "qos_monitoring_normal": {"train": 2_450, "validation": 525, "test": 525},
}

ABNORMAL_EVENTS = (
    "authentication_failure",
    "registration_rejection",
    "nrf_discovery_failure",
    "packet_loss",
    "resource_exhaustion",
    "excessive_latency",
    "handover_failure",
    "signalling_storm",
    "malformed_message",
    "slice_policy_violation",
)

for index, event_name in enumerate(ABNORMAL_EVENTS):
    validation_count = 188 if index < 5 else 187
    EVENT_SPLIT_COUNTS[event_name] = {
        "train": 875,
        "validation": validation_count,
        "test": 375 - validation_count,
    }

NORMAL_EVENTS = tuple(name for name in EVENT_SPLIT_COUNTS if name not in ABNORMAL_EVENTS)
EVENT_NUMBER = {name: index for index, name in enumerate(EVENT_SPECS, start=1)}


def partition_count(total: int, target_size: int = 6) -> list[int]:
    """Split a row quota into compact scenario sizes without losing rows."""

    scenario_count = max(1, round(total / target_size))
    base_size, remainder = divmod(total, scenario_count)
    sizes = [base_size + 1] * remainder + [base_size] * (scenario_count - remainder)
    if min(sizes) < 3 or max(sizes) > 9:
        raise ValueError(f"Unexpected scenario-size range for quota {total}: {min(sizes)}-{max(sizes)}")
    return sizes


def build_scenario_plans(rng: random.Random) -> list[ScenarioPlan]:
    """Create scenario groups with exact event and split row totals."""

    plans: list[ScenarioPlan] = []
    for event_name, split_counts in EVENT_SPLIT_COUNTS.items():
        label = "abnormal" if event_name in ABNORMAL_EVENTS else "normal"
        for split, row_count in split_counts.items():
            sizes = partition_count(row_count)
            rng.shuffle(sizes)
            plans.extend(ScenarioPlan(split, label, event_name, size) for size in sizes)
    rng.shuffle(plans)
    return plans


def build_severity_pools(rng: random.Random) -> dict[tuple[str, str], list[str]]:
    """Build shuffled pools so every severity quota is satisfied exactly."""

    pools: dict[tuple[str, str], list[str]] = {}
    for key, quota in SEVERITY_QUOTAS.items():
        values = [severity for severity, count in quota.items() for _ in range(count)]
        rng.shuffle(values)
        pools[key] = values
    return pools


def make_context(rng: random.Random, event_name: str, session_id: str) -> dict[str, object]:
    """Generate valid-looking parameters shared by every row in a scenario."""

    is_latency_event = event_name == "excessive_latency"
    is_resource_event = event_name == "resource_exhaustion"
    return {
        "supi": f"supi-00101{rng.randrange(10_000_000_000):010d}",
        "guti": f"guti-{rng.randrange(16**8):08X}",
        "gnb": f"gNB-{rng.randint(1, 180):03d}",
        "source_gnb": f"gNB-{rng.randint(1, 180):03d}",
        "target_gnb": f"gNB-{rng.randint(181, 360):03d}",
        "cell_id": f"CELL-{rng.randint(1, 9_999):04d}",
        "target_cell_id": f"CELL-{rng.randint(10_000, 19_999):05d}",
        "session_id": session_id,
        "slice_id": rng.choice(("sst-1-sd-010203", "sst-1-sd-112233", "sst-2-sd-445566", "sst-3-sd-778899")),
        "allowed_slice_id": rng.choice(("sst-1-sd-010203", "sst-2-sd-445566")),
        "algorithm": rng.choice(("128-NIA1", "128-NIA2", "128-NIA3")),
        "security_context": f"SEC-{rng.randint(10_000, 99_999)}",
        "dnn": rng.choice(("internet", "enterprise", "iot", "ims")),
        "policy_id": f"POL-{rng.randint(1_000, 9_999)}",
        "policy_version": rng.randint(2, 24),
        "teid": rng.randint(100_000, 9_999_999),
        "qfi": rng.randint(1, 63),
        "qos_profile": rng.choice(("5QI-1", "5QI-7", "5QI-8", "5QI-9")),
        "ue_ip": f"10.{rng.randint(10, 240)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}",
        "service_ip": f"10.{rng.randint(10, 240)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}:{rng.choice((80, 443, 7777))}",
        "nf_instance": f"nf-{rng.randint(1_000, 9_999)}",
        "service_name": rng.choice(("namf-comm", "nsmf-pdusession", "nudm-sdm", "npcf-smpolicycontrol")),
        "target_nf": rng.choice(("AMF", "SMF", "UDM", "AUSF", "PCF")),
        "requester_nf": rng.choice(("AMF", "SMF", "SCP", "NSSF")),
        "validity_seconds": rng.choice((300, 600, 900, 1_800, 3_600)),
        "priority": rng.randint(1, 10),
        "capacity_pct": rng.randint(90, 99) if is_resource_event else rng.randint(35, 82),
        "heartbeat_seconds": rng.choice((10, 15, 30, 60)),
        "success_count": rng.randint(4, 180),
        "usage_limit_mb": rng.choice((512, 1_024, 2_048, 5_120, 10_240)),
        "throughput_mbps": rng.randint(80, 950),
        "degraded_throughput_mbps": rng.randint(1, 24),
        "low_loss_pct": round(rng.uniform(0.01, 0.9), 2),
        "packet_loss_pct": rng.randint(15, 82),
        "packet_count": rng.randint(800, 80_000),
        "low_queue_pct": rng.randint(8, 48),
        "high_queue_pct": rng.randint(91, 100),
        "memory_pct": rng.randint(90, 99),
        "queue_depth": rng.randint(250, 9_500),
        "monitor_seconds": rng.choice((30, 60, 120, 300, 600)),
        "retry_count": rng.randint(3, 12),
        "latency_ms": rng.randint(220, 2_500) if is_latency_event else rng.randint(5, 95),
        "latency_threshold_ms": rng.choice((20, 50, 100, 150)),
        "signal_dbm": rng.randint(-88, -60),
        "weak_signal_dbm": rng.randint(-128, -106),
        "cause_code": rng.choice(("ILLEGAL_UE", "SLICE_NOT_AVAILABLE", "CONGESTION", "PROTOCOL_ERROR")),
        "http_status": rng.choice((408, 429, 500, 503, 504)),
        "request_rate": rng.randint(1_500, 12_000),
        "rate_threshold": rng.choice((500, 750, 1_000, 1_250)),
        "ue_group": f"UEG-{rng.randint(100, 999)}",
        "invalid_field": rng.choice(("procedureCode", "sessionType", "securityHeader", "sliceIdentifier")),
        "byte_offset": rng.randint(12, 1_400),
        "protocol_version": f"{rng.randint(6, 9)}.{rng.randint(0, 9)}",
        "validation_errors": rng.randint(2, 14),
    }


def format_timestamp(value: datetime) -> tuple[str, str]:
    """Return a machine-readable timestamp and the form embedded in log text."""

    iso_value = value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    text_value = value.strftime("%Y-%m-%d %H:%M:%S.") + f"{value.microsecond // 1_000:03d}"
    return iso_value, text_value


def make_row(
    rng: random.Random,
    plan: ScenarioPlan,
    context: dict[str, object],
    scenario_id: str,
    session_id: str,
    severity: str,
    timestamp: datetime,
) -> dict[str, str]:
    """Render one row from a selected pattern and shared scenario context."""

    event = EVENT_SPECS[plan.event_name]
    message_index = rng.randrange(len(event.templates))

    # The fourth detail creates templates that occur only in a subset of test rows.
    if plan.split == "test" and rng.random() < 0.25:
        detail_index = 3
    else:
        detail_index = rng.randrange(3)

    network_function, message_pattern = event.templates[message_index]
    message = message_pattern.format(**context)
    detail = event.details[detail_index].format(**context)
    iso_timestamp, text_timestamp = format_timestamp(timestamp)
    template_id = f"TPL-{EVENT_NUMBER[plan.event_name]:03d}-{message_index + 1:02d}-{detail_index + 1:02d}"

    return {
        "timestamp": iso_timestamp,
        "network_function": network_function,
        "severity": severity,
        "log_text": f"{text_timestamp} [{network_function}] {severity} {message}; {detail}",
        "label": plan.label,
        "anomaly_type": plan.event_name if plan.label == "abnormal" else "none",
        "scenario_id": scenario_id,
        "session_id": session_id,
        "template_id": template_id,
        "split": plan.split,
    }


def generate_rows(seed: int) -> list[dict[str, str]]:
    """Generate rows in chronological scenario order using a reproducible seed."""

    rng = random.Random(seed)
    plans = build_scenario_plans(rng)
    severity_pools = build_severity_pools(rng)
    rows: list[dict[str, str]] = []
    clock = datetime(2026, 1, 1, tzinfo=UTC)

    for scenario_number, plan in enumerate(plans, start=1):
        scenario_id = f"SCN-{scenario_number:06d}"
        session_id = f"SES-{scenario_number:06d}"
        context = make_context(rng, plan.event_name, session_id)

        # Related rows share identifiers and are separated by realistic millisecond gaps.
        clock += timedelta(milliseconds=rng.randint(500, 20_000))
        pool = severity_pools[(plan.split, plan.label)]
        for _ in range(plan.size):
            clock += timedelta(milliseconds=rng.randint(15, 1_500))
            rows.append(
                make_row(
                    rng=rng,
                    plan=plan,
                    context=context,
                    scenario_id=scenario_id,
                    session_id=session_id,
                    severity=pool.pop(),
                    timestamp=clock,
                )
            )

    if any(severity_pools.values()):
        remaining = {key: len(values) for key, values in severity_pools.items() if values}
        raise AssertionError(f"Unused severity quotas: {remaining}")
    return rows


def validate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    """Fail fast if the dataset violates any agreed research requirement."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    require(len(rows) == TOTAL_LOGS, f"Expected {TOTAL_LOGS} rows, found {len(rows)}")
    require(all(set(row) == set(FIELDNAMES) for row in rows), "One or more rows have an invalid schema")
    require(all(all(str(row[field]).strip() for field in FIELDNAMES) for row in rows), "Empty values found")

    class_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    split_label_counts = Counter((row["split"], row["label"]) for row in rows)
    severity_counts = Counter((row["label"], row["severity"]) for row in rows)
    anomaly_counts = Counter(row["anomaly_type"] for row in rows if row["label"] == "abnormal")

    require(dict(class_counts) == CLASS_COUNTS, f"Incorrect class counts: {class_counts}")
    require(dict(split_counts) == SPLIT_COUNTS, f"Incorrect split counts: {split_counts}")
    require(dict(split_label_counts) == SPLIT_LABEL_COUNTS, f"Incorrect split/label counts: {split_label_counts}")

    expected_severity = Counter()
    for (_, label), quota in SEVERITY_QUOTAS.items():
        for severity, count in quota.items():
            expected_severity[(label, severity)] += count
    require(severity_counts == expected_severity, f"Incorrect severity counts: {severity_counts}")
    require(all(anomaly_counts[name] == 1_250 for name in ABNORMAL_EVENTS), f"Unbalanced anomaly types: {anomaly_counts}")
    require(all(row["anomaly_type"] == "none" for row in rows if row["label"] == "normal"), "Normal row has anomaly type")

    scenario_splits: dict[str, set[str]] = defaultdict(set)
    session_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scenario_splits[row["scenario_id"]].add(row["split"])
        session_splits[row["session_id"]].add(row["split"])
    require(all(len(values) == 1 for values in scenario_splits.values()), "A scenario crosses dataset splits")
    require(all(len(values) == 1 for values in session_splits.values()), "A session crosses dataset splits")

    duplicate_log_count = len(rows) - len({row["log_text"] for row in rows})
    require(duplicate_log_count / len(rows) < 0.03, "Exact log-text duplicate rate exceeds 3%")
    require(
        not any(re.search(r"\b(?:normal|abnormal)\b", row["log_text"], re.IGNORECASE) for row in rows),
        "A target-label word appears inside log text",
    )

    train_validation_templates = {
        row["template_id"] for row in rows if row["split"] in {"train", "validation"}
    }
    test_templates = {row["template_id"] for row in rows if row["split"] == "test"}
    test_only_templates = test_templates - train_validation_templates
    require(test_only_templates, "No test-only template variants were generated")

    token_counts = [len(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", row["log_text"])) for row in rows]
    average_tokens = sum(token_counts) / len(token_counts)
    require(10 <= average_tokens <= 30, f"Unexpected average token count: {average_tokens:.2f}")

    return {
        "rows": len(rows),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "severity_counts": {f"{label}/{severity}": count for (label, severity), count in sorted(severity_counts.items())},
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "scenarios": len(scenario_splits),
        "sessions": len(session_splits),
        "unique_templates": len({row["template_id"] for row in rows}),
        "test_only_templates": len(test_only_templates),
        "exact_duplicate_logs": duplicate_log_count,
        "average_tokens_per_log": round(average_tokens, 2),
    }


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write a flat UTF-8 CSV with one observation per row."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("synthetic_5g_network_logs_50000.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output, help="Destination CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Reproducible random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_rows(args.seed)
    report = validate_rows(rows)
    write_csv(rows, args.output)

    print(f"Created: {args.output.resolve()}")
    print(f"Rows: {report['rows']:,}")
    print(f"Classes: {report['class_counts']}")
    print(f"Splits: {report['split_counts']}")
    print(f"Severity: {report['severity_counts']}")
    print(f"Scenarios/sessions: {report['scenarios']:,}/{report['sessions']:,}")
    print(f"Templates: {report['unique_templates']} ({report['test_only_templates']} test-only)")
    print(f"Exact duplicate logs: {report['exact_duplicate_logs']}")
    print(f"Average tokens per log: {report['average_tokens_per_log']}")


if __name__ == "__main__":
    main()
