"""Read log files, reproduce notebook cleaning, and keep research splits separate."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


# These expressions match 01_data_cleaning_preprocessing.ipynb exactly.
LEADING_TIMESTAMP_RE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*"
)
REPLACEMENTS = [
    (re.compile(r"\bsupi-\d+\b", re.I), "supi_id"),
    (re.compile(r"\bguti-[0-9a-f]+\b", re.I), "guti_id"),
    (re.compile(r"\bses-\d+\b", re.I), "session_id"),
    (re.compile(r"\bgnb-\d+\b", re.I), "gnb_id"),
    (re.compile(r"\bcell-\d+\b", re.I), "cell_id"),
    (re.compile(r"\bnf-\d+\b", re.I), "nf_instance_id"),
    (re.compile(r"\bpol-\d+\b", re.I), "policy_id"),
    (re.compile(r"\bsec-\d+\b", re.I), "security_context_id"),
    (re.compile(r"\bueg-\d+\b", re.I), "ue_group_id"),
    (re.compile(r"\bsst-\d+-sd-[0-9a-f]+\b", re.I), "slice_id"),
    (re.compile(r"\b(?:10|172|192)\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "ip_address"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "num_value"),
]
PARTITIONS = ("train", "validation", "test")
TEXT_COLUMNS = ("clean_text", "log_text", "message", "log_message", "log", "text")


def clean_log_text(value: Any) -> str:
    """Apply the supplied preprocessing notebook's conservative normalisation."""
    text = LEADING_TIMESTAMP_RE.sub("", str(value)).lower()
    for pattern, replacement in REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^a-z0-9_\-\s]", " ", text)
    text = text.replace("-", "_")
    return re.sub(r"\s+", " ", text).strip()


def _blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype(str).str.strip().eq("")


def _column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(name).strip().lower(): name for name in frame.columns}
    return next((lookup[name] for name in names if name in lookup), None)


def _strings(frame: pd.DataFrame, name: str | None) -> pd.Series:
    if name is None:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[name].fillna("").astype(str).str.strip()


def parse_file(path_or_filelike: Any, filename: str | None = None) -> pd.DataFrame:
    """Read CSV, one-message-per-line TXT, or a JSON log list without executing it."""
    name = filename or getattr(path_or_filelike, "name", str(path_or_filelike))
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".txt", ".json"}:
        raise ValueError("Choose a CSV, TXT, or JSON file.")
    if isinstance(path_or_filelike, (str, Path)):
        content = Path(path_or_filelike).read_bytes()
    elif isinstance(path_or_filelike, (bytes, bytearray)):
        content = bytes(path_or_filelike)
    else:
        content = path_or_filelike.read()
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("The log file must use UTF-8 text encoding.") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The uploaded file is empty.")

    # Keep textual IDs and original cell values intact when reading CSV files.
    if suffix == ".csv":
        try:
            return pd.read_csv(io.StringIO(content), dtype="string", keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise ValueError(f"The CSV could not be read: {error}") from error
    if suffix == ".txt":
        return pd.DataFrame({"log_text": [line for line in content.splitlines() if line.strip()]})
    try:
        records = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"The JSON could not be read: {error.msg}") from error
    if isinstance(records, dict):
        records = records.get("logs")
    if not isinstance(records, list) or not records:
        raise ValueError("JSON must contain a nonempty list of logs or an object with a 'logs' list.")
    if all(isinstance(record, str) for record in records):
        return pd.DataFrame({"log_text": records})
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("JSON logs must all be message strings or all be objects.")
    if any(isinstance(value, (dict, list)) for record in records for value in record.values()):
        raise ValueError("Each JSON log field must contain a scalar value.")
    return pd.DataFrame(records)


def _binary(values: pd.Series, name: str) -> pd.Series:
    # Numeric labels must equal zero or one; decimals are never silently rounded.
    words = {"normal": 0, "benign": 0, "abnormal": 1, "anomalous": 1, "anomaly": 1}
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for index, value in values.items():
        if pd.isna(value) or str(value).strip() == "":
            continue
        token = str(value).strip().lower()
        if token in words:
            result.loc[index] = words[token]
            continue
        try:
            number = float(token)
        except (TypeError, ValueError):
            number = np.nan
        if not np.isfinite(number) or number not in (0.0, 1.0):
            raise ValueError(f"Column '{name}' contains an invalid label at original row {index + 1}; use normal/abnormal or exactly 0/1.")
        result.loc[index] = number
    return result


def _group_leakage(frame: pd.DataFrame) -> dict[str, int]:
    result = {}
    for field in ("scenario_id", "session_id"):
        column = _column(frame, (field,))
        if column is None:
            result[field] = 0
            continue
        values = _strings(frame, column)
        known = values.ne("") & frame["_split"].ne("")
        grouped = pd.DataFrame({"group": values[known], "split": frame.loc[known, "_split"]})
        result[field] = int(grouped.groupby("group")["split"].nunique().gt(1).sum())
    return result


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.value_counts().items() if str(key).strip()}


def prepare_data(raw: pd.DataFrame, *, source_name: str = "uploaded", is_default: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preserve uploaded records while adding explicit modelling and audit fields."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("The dataset contains no log records.")
    if raw.columns.duplicated().any():
        raise ValueError("The dataset has repeated column names.")
    reserved = {"_text", "_target", "_session", "_template", "_timestamp", "_split"}
    if reserved.intersection(raw.columns):
        raise ValueError("Columns beginning with the reserved modelling names are not valid upload fields.")
    frame = raw.copy().reset_index(drop=True)
    original_columns = list(frame.columns)
    text_column = _column(frame, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError("A clean_text, log_text, message, log_message, log, or text column is required.")

    # Remove only exact original duplicates; IDs retain their original file positions.
    duplicate_mask = frame.duplicated()
    missing = {str(column): int(_blank(frame[column]).sum()) for column in frame.columns}
    if "row_id" in frame:
        original_id = "original_row_id"
        while original_id in frame:
            original_id = "original_" + original_id
        frame = frame.rename(columns={"row_id": original_id})
    frame["row_id"] = np.arange(1, len(frame) + 1, dtype=int)
    frame = frame.loc[~duplicate_mask].copy()
    if _blank(frame[text_column]).any():
        raise ValueError("Every log record must have nonempty text; blank messages are not silently discarded.")
    if str(text_column).strip().lower() == "clean_text":
        frame["_text"] = frame[text_column].astype(str)
    else:
        frame["_text"] = frame[text_column].map(clean_log_text)
    if frame["_text"].str.strip().eq("").any():
        raise ValueError("Cleaning produced an empty log message.")

    # Labels are optional for prediction but must agree whenever both are supplied.
    label_columns = [column for column in (_column(frame, ("target",)), _column(frame, ("label",))) if column is not None]
    if not label_columns:
        fallback = _column(frame, ("is_anomaly", "anomaly", "class"))
        label_columns = [fallback] if fallback is not None else []
    targets = [_binary(frame[column], str(column)) for column in label_columns]
    frame["_target"] = targets[0] if targets else np.nan
    for other in targets[1:]:
        known = frame["_target"].notna() & other.notna()
        if frame.loc[known, "_target"].ne(other[known]).any():
            raise ValueError("The label and target columns disagree on one or more log records.")
        frame["_target"] = frame["_target"].fillna(other)
    raw_text_column = _column(frame, ("log_text",)) or text_column
    # Provide one display field while retaining the uploaded source column.
    if "log_text" not in frame:
        frame["log_text"] = frame[raw_text_column].astype(str)
    labelled = frame[frame["_target"].notna()]
    if labelled.groupby(raw_text_column)["_target"].nunique().gt(1).any():
        raise ValueError("Identical log messages contain contradictory labels.")

    # Genuine IDs stay blank when absent; generated templates are clearly recorded.
    session_column = _column(frame, ("session_id",))
    template_column = _column(frame, ("template_id",))
    timestamp_column = _column(frame, ("timestamp", "time", "datetime"))
    frame["_session"] = _strings(frame, session_column)
    frame["_template"] = _strings(frame, template_column)
    generated_templates = frame["_template"].eq("")
    frame.loc[generated_templates, "_template"] = frame.loc[generated_templates, "_text"].map(
        lambda text: "GENERATED-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    )
    frame["_timestamp"] = pd.to_datetime(_strings(frame, timestamp_column), errors="coerce", utc=True, format="mixed")
    split_column = _column(frame, ("split", "dataset_split", "partition"))
    frame["_split"] = _strings(frame, split_column).str.lower().replace({"val": "validation", "valid": "validation", "dev": "validation"})
    invalid_splits = sorted(set(frame["_split"]) - {"", *PARTITIONS})
    if invalid_splits:
        raise ValueError(f"Unrecognised split values: {', '.join(invalid_splits)}. Use train, validation, or test.")

    # Session experiments require real complete identifiers and coherent session labels.
    session_conflicts = int(frame.loc[frame["_session"].ne("")].groupby("_session")["_target"].nunique().gt(1).sum())
    can_sessions = bool(session_column is not None and template_column is not None
                        and frame["_session"].ne("").all() and not generated_templates.any()
                        and session_conflicts == 0)
    train, test = (frame.loc[frame["_split"].eq(name)] for name in ("train", "test"))
    real_sessions = frame.loc[frame["_session"].ne(""), "_session"]
    network_column = _column(frame, ("network_function",))
    anomaly_column = _column(frame, ("anomaly_type",))
    context = {
        "name": source_name, "row_count": int(len(frame)),
        "label_counts": {"normal": int(frame["_target"].eq(0).sum()), "abnormal": int(frame["_target"].eq(1).sum())},
        "split_counts": _counts(frame["_split"]),
        "network_function_counts": _counts(_strings(frame, network_column)),
        "anomaly_type_counts": _counts(_strings(frame, anomaly_column)),
        "session_count": int(real_sessions.nunique()), "template_count": int(frame["_template"].nunique()),
        "unique_clean_texts": int(frame["_text"].nunique()), "columns": [str(column) for column in original_columns],
        "source": "Synthetic 5G research dataset supplied with this prototype" if is_default else "User-provided dataset",
        "has_labels": bool(frame["_target"].notna().all()), "can_sessions": can_sessions,
        "text_column": str(text_column), "generated_templates": int(generated_templates.sum()),
        "unlabelled_rows": int(frame["_target"].isna().sum()),
        "audit": {"missing": missing, "duplicates": int(duplicate_mask.sum()),
                  "repeatedclean": int(frame["_text"].duplicated().sum()),
                  "train_test_text_overlap": len(set(train["_text"]) & set(test["_text"])),
                  "template_overlap": len(set(train["_template"]) & set(test["_template"])),
                  "group_leakage": _group_leakage(frame), "session_label_conflicts": session_conflicts,
                  "invalid_or_missing_timestamps": int(frame["_timestamp"].isna().sum())},
    }
    return frame.reset_index(drop=True), context


def _connected_groups(frame: pd.DataFrame) -> np.ndarray:
    """Join records sharing either scenario or session IDs, including transitive links."""
    parent = np.arange(len(frame))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for field in ("scenario_id", "session_id"):
        column = _column(frame, (field,))
        if column is None:
            continue
        seen: dict[str, int] = {}
        for index, value in enumerate(_strings(frame, column)):
            if not value:
                continue
            root = find(index)
            if value in seen:
                parent[root] = find(seen[value])
            else:
                seen[value] = root
    return np.array([find(index) for index in range(len(frame))])


def _validate_partitions(frame: pd.DataFrame) -> None:
    if frame["_target"].isna().any():
        raise ValueError("Training requires a binary label for every record.")
    if set(frame["_split"]) != set(PARTITIONS):
        raise ValueError("Training needs nonempty train, validation, and test partitions for every row.")
    for name in PARTITIONS:
        if set(frame.loc[frame["_split"].eq(name), "_target"].unique()) != {0, 1}:
            raise ValueError(f"The {name} partition must contain both normal and abnormal records.")
    if any(_group_leakage(frame).values()):
        raise ValueError("The provided split shares scenario or session IDs across partitions. Choose a custom grouped split or correct the input.")


def apply_split(frame: pd.DataFrame, mode: str = "provided", test_size: float = 0.15,
                validation_size: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Validate supplied splits or allocate whole connected groups deterministically."""
    result = frame.copy().reset_index(drop=True)
    if mode == "provided":
        _validate_partitions(result)
        return result
    if mode != "custom":
        raise ValueError("Split mode must be 'provided' or 'custom'.")
    if not (0 < test_size < 1 and 0 < validation_size < 1 and test_size + validation_size < 1):
        raise ValueError("Test and validation fractions must be positive and leave some training data.")
    if result["_target"].isna().any() or set(result["_target"].unique()) != {0, 1}:
        raise ValueError("Training requires labelled records from both normal and abnormal classes.")

    # Every connected group is indivisible, preventing both direct and transitive leakage.
    groups = _connected_groups(result)
    group_table = pd.crosstab(groups, result["_target"]).reindex(columns=[0, 1], fill_value=0)
    counts = group_table.to_numpy(dtype=float)
    group_ids = group_table.index.to_numpy()
    pure_normal = np.flatnonzero((counts[:, 0] > 0) & (counts[:, 1] == 0))
    pure_abnormal = np.flatnonzero((counts[:, 1] > 0) & (counts[:, 0] == 0))
    mixed = np.flatnonzero((counts > 0).all(axis=1))
    mixed_needed = min(3, len(mixed))
    if len(pure_normal) < 3 - mixed_needed or len(pure_abnormal) < 3 - mixed_needed:
        raise ValueError("There are too few independent scenario/session groups to put both classes in all three partitions.")
    fractions = np.array([1 - validation_size - test_size, validation_size, test_size])
    targets = fractions[:, None] * counts.sum(axis=0)[None, :]
    rng = np.random.default_rng(seed)
    best_assignments, best_score = None, float("inf")

    # Seed class coverage first, then minimise class-count deviations from requested sizes.
    for _ in range(6):
        assignments = np.full(len(group_ids), -1, dtype=int)
        totals = np.zeros((3, 2), dtype=float)
        mixed_order = rng.permutation(mixed)
        normal_order = rng.permutation(pure_normal)
        abnormal_order = rng.permutation(pure_abnormal)
        for partition in range(3):
            chosen = [mixed_order[partition]] if partition < mixed_needed else [normal_order[partition - mixed_needed], abnormal_order[partition - mixed_needed]]
            for group in chosen:
                assignments[group] = partition
                totals[partition] += counts[group]
        pending = rng.permutation(np.flatnonzero(assignments < 0))
        pending = pending[np.argsort(-counts[pending].sum(axis=1), kind="stable")]
        for group in pending:
            before = ((totals - targets) ** 2 / np.maximum(targets, 1)).sum(axis=1)
            after = ((totals + counts[group] - targets) ** 2 / np.maximum(targets, 1)).sum(axis=1)
            partition = int(np.argmin(after - before))
            assignments[group] = partition
            totals[partition] += counts[group]
        score = float(((totals - targets) ** 2 / np.maximum(targets, 1)).sum())
        if score < best_score:
            best_score, best_assignments = score, assignments.copy()
    allocation = {group: PARTITIONS[int(partition)] for group, partition in zip(group_ids, best_assignments)}
    result["_split"] = [allocation[group] for group in groups]
    _validate_partitions(result)
    return result
