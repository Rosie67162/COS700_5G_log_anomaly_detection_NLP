"""Train and retain controlled log-entry and session experiments."""

from __future__ import annotations

# Import numeric tools and the notebook's estimators.
import copy
import random
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# Support Gensim releases that still import this SciPy helper.
if not hasattr(scipy.linalg, "triu"):
    scipy.linalg.triu = np.triu

import torch
from gensim.models import Word2Vec
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


# Keep representation identifiers stable across saved runs.
REPRESENTATIONS = ("tfidf", "embedding", "concat", "weighted", "fusion")
CLASSIFIERS = {"lr": "Logistic Regression", "rf": "Random Forest"}
DEFAULTS = {"group": "logs", "representation": "all", "classifier": "all", "seed": 42,
            "rf_trees": 120, "w2v_epochs": 5, "logbert_epochs": 5, "word2vec_dim": 100,
            "logbert_dim": 64, "max_features": 20000, "batch_size": 256}
PAD_ID, DIST_ID, MASK_ID, UNK_ID = 0, 1, 2, 3


def representation_name(code: str, group: str) -> str:
    # Label each method within its evaluation group.
    embedding = "Word2Vec" if group == "logs" else "LogBERT"
    return {"tfidf": "TF-IDF", "embedding": embedding,
            "concat": f"TF-IDF + {embedding} concatenation",
            "weighted": f"TF-IDF-weighted {embedding}",
            "fusion": f"Decision fusion: TF-IDF + {embedding}"}[code]


def _settings(config: dict) -> dict:
    # Reject invalid settings before expensive processing starts.
    options = {**DEFAULTS, **config}
    if options["group"] not in {"logs", "sessions"}:
        raise ValueError("Choose the log-entry or session experiment group.")
    if options["representation"] not in (*REPRESENTATIONS, "all"):
        raise ValueError("Unknown representation.")
    if options["classifier"] not in (*CLASSIFIERS, "all"):
        raise ValueError("Unknown classifier.")
    for key in ("seed", "rf_trees", "w2v_epochs", "logbert_epochs", "word2vec_dim",
                "logbert_dim", "max_features", "batch_size"):
        options[key] = int(options[key])
        if options[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be positive.")
    if options["logbert_dim"] % 4:
        raise ValueError("The LogBERT dimension must be divisible by four attention heads.")
    return options


def _seed(seed: int) -> None:
    # Repeat the notebook's deterministic training setup.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _outcome(actual: int, predicted: int) -> str:
    # Convert a labelled prediction into its confusion-matrix outcome.
    return {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(actual, predicted)]


def _make_units(frame: pd.DataFrame, group: str, *, training: bool) -> pd.DataFrame:
    # Preserve complete sessions and source records during aggregation.
    required = {"_text"} | ({"_target", "_split"} if training else set())
    if group == "sessions":
        required |= {"_session", "_template"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing prepared columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("There are no records to process.")
    source = frame.copy()
    if "row_id" not in source:
        source["row_id"] = [str(i) for i in source.index]
    if "log_text" not in source:
        source["log_text"] = source["_text"]
    anomaly_column = next((c for c in ("_anomaly_type", "anomaly_type") if c in source), None)
    if group == "logs":
        source["unit_id"] = source["row_id"].astype(str)
        source["session_id"] = source.get("_session", pd.Series("", index=source.index)).astype(str)
        source["known_anomaly_type"] = source[anomaly_column].fillna("").astype(str) if anomaly_column else ""
        source["event_count"] = 1
        return source.reset_index(drop=True)
    if any(source[column].isna().any() or source[column].astype(str).str.strip().eq("").any()
           for column in ("_session", "_template")):
        raise ValueError("Session models require a genuine session ID and event-template ID for every record.")
    records = []
    for session_id, rows in source.groupby("_session", sort=False):
        if "_timestamp" in rows:
            rows = rows.sort_values("_timestamp", kind="stable")
        if training and (rows["_split"].nunique() != 1 or rows["_target"].nunique() != 1):
            raise ValueError(f"Session {session_id} crosses partitions or labels; correct the session data first.")
        entry = {"unit_id": str(session_id), "session_id": str(session_id),
                 "row_id": str(rows["row_id"].iloc[0]), "row_ids": rows["row_id"].astype(str).tolist(),
                 "_text": " [EVENT] ".join(rows["_text"].astype(str)),
                 "log_text": "\n".join(rows["log_text"].astype(str)),
                 "log_keys": rows["_template"].astype(str).tolist(), "event_count": len(rows),
                 "known_anomaly_type": str(rows[anomaly_column].iloc[0]) if anomaly_column else ""}
        if "_target" in rows:
            entry["_target"] = int(rows["_target"].max()) if rows["_target"].isin([0, 1]).all() else np.nan
        if "_split" in rows:
            entry["_split"] = str(rows["_split"].iloc[0])
        records.append(entry)
    return pd.DataFrame(records)


def _threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    # Select the maximum-F1 threshold using validation labels only.
    precision, recall, values = precision_recall_curve(target, probability)
    scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(scores))
    return float(values[best]), float(scores[best])


def _metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    # Derive every metric from the same held-out predictions.
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {"accuracy": float(accuracy_score(target, prediction)),
            "precision": float(precision_score(target, prediction, zero_division=0)),
            "recall": float(recall_score(target, prediction, zero_division=0)),
            "f1": float(f1_score(target, prediction, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "roc_auc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
            "pr_auc": float(average_precision_score(target, probability)) if np.any(target == 1) else None,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion_matrix": matrix.tolist(),
            "test_count": len(target)}


class MaskedSessions(Dataset):
    # Mask event identifiers for normal-session self-supervision.
    def __init__(self, sequences: list[list[int]], masked: bool = True):
        self.sequences, self.masked = sequences, masked

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        values = torch.tensor([DIST_ID] + self.sequences[index], dtype=torch.long)
        labels = torch.full_like(values, -100)
        if self.masked:
            positions = torch.arange(1, len(values))
            selected = positions[torch.randperm(len(positions))[:max(1, round(len(positions) * 0.3))]]
            labels[selected] = values[selected]
            values[selected] = MASK_ID
        return values, labels


def _collate(batch):
    # Pad variable-length sessions without training on padding.
    values, labels = zip(*batch)
    values = pad_sequence(values, batch_first=True, padding_value=PAD_ID)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return values, labels, values.eq(PAD_ID)


class LogBERTEncoder(nn.Module):
    # Match the notebook's two-layer custom LogBERT-style encoder.
    def __init__(self, vocabulary_size: int, max_length: int, dimension: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, dimension, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, dimension)
        layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=4, dim_feedforward=256,
                                           dropout=0.1, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(dimension)
        self.mlm_head = nn.Linear(dimension, vocabulary_size)

    def forward(self, values, padding):
        # Add event positions before contextual encoding.
        positions = torch.arange(values.size(1), device=values.device).unsqueeze(0)
        hidden = self.token_embedding(values) + self.position_embedding(positions)
        hidden = self.output_norm(self.encoder(hidden, src_key_padding_mask=padding))
        return self.mlm_head(hidden), hidden


class FeatureSpace:
    # Retain training-fitted feature builders for future predictions.
    def __init__(self, config: dict, representations: list[str]):
        self.config = config
        self.group = config["group"]
        self.representations = representations
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=config["max_features"],
                                     sublinear_tf=True, dtype=np.float32)
        self.mean_scaler, self.weighted_scaler = StandardScaler(), StandardScaler()
        self.w2v = None
        self.encoder = None
        self.history = []
        self.training_seconds = 0.0
        self.vocabulary = {}
        self.idf = {}
        self.key_idf = {}
        self.max_events = 0
        self.unknown_weight = 1.0
        self.requires_embedding = any(r != "tfidf" for r in representations)

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame, progress: Callable) -> dict:
        # Fit vocabularies and scalers exclusively on training units.
        started = time.perf_counter()
        lexical = self.tfidf.fit_transform(train["_text"])
        self.idf = dict(zip(self.tfidf.get_feature_names_out(), self.tfidf.idf_))
        if not self.requires_embedding:
            self.training_seconds = time.perf_counter() - started
            return {"tfidf": lexical}
        if self.group == "logs":
            progress(12, "Learning Word2Vec from training logs")
            self.w2v = Word2Vec(sentences=[x.split() for x in train["_text"]],
                                vector_size=self.config["word2vec_dim"], window=5, min_count=2,
                                workers=1, sg=1, seed=self.config["seed"], epochs=self.config["w2v_epochs"])
        else:
            self._fit_encoder(train, validation, progress)
        mean_raw, weighted_raw = self._raw_embeddings(train)
        mean = self.mean_scaler.fit_transform(mean_raw).astype(np.float32)
        weighted = self.weighted_scaler.fit_transform(weighted_raw).astype(np.float32)
        self.training_seconds = time.perf_counter() - started
        return self._assemble(lexical, mean, weighted)

    def transform(self, units: pd.DataFrame) -> dict:
        # Apply the saved training transforms without refitting.
        lexical = self.tfidf.transform(units["_text"])
        if not self.requires_embedding:
            return {"tfidf": lexical}
        mean_raw, weighted_raw = self._raw_embeddings(units)
        mean = self.mean_scaler.transform(mean_raw).astype(np.float32)
        weighted = self.weighted_scaler.transform(weighted_raw).astype(np.float32)
        return self._assemble(lexical, mean, weighted)

    def _assemble(self, lexical, mean, weighted) -> dict:
        # Allocate only the feature matrices required by the selected methods.
        result = {"tfidf": lexical, "embedding": mean, "weighted": weighted}
        if "concat" in self.representations:
            result["concat"] = hstack([lexical, csr_matrix(mean)], format="csr")
        return result

    def _raw_embeddings(self, units: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Reuse repeated log vectors to keep large datasets responsive.
        if self.group == "sessions":
            return self._session_embeddings(units["log_keys"].tolist())
        unique, indices = np.unique(units["_text"].to_numpy(dtype=str), return_inverse=True)
        dimension = self.config["word2vec_dim"]
        mean_rows, weighted_rows = [], []
        for text in unique:
            tokens = text.split()
            present = [token for token in tokens if token in self.w2v.wv]
            mean_rows.append(np.mean([self.w2v.wv[token] for token in present], axis=0)
                             if present else np.zeros(dimension, dtype=np.float32))
            weighted_tokens = [token for token in present if token in self.idf]
            weights = [self.idf[token] for token in weighted_tokens]
            weighted_rows.append(np.average([self.w2v.wv[token] for token in weighted_tokens],
                                             weights=weights, axis=0)
                                 if sum(weights) > 0 else np.zeros(dimension, dtype=np.float32))
        return (np.asarray(mean_rows, dtype=np.float32)[indices],
                np.asarray(weighted_rows, dtype=np.float32)[indices])

    def _map_keys(self, keys: list[str]) -> list[int]:
        # Map unseen events to the saved unknown-event identifier.
        return [self.vocabulary.get(key, UNK_ID) for key in keys[:self.max_events]]

    def _fit_encoder(self, train: pd.DataFrame, validation: pd.DataFrame, progress: Callable) -> None:
        # Train masked-event and compactness objectives on normal sessions.
        self.vocabulary = {"[PAD]": PAD_ID, "[DIST]": DIST_ID, "[MASK]": MASK_ID, "[UNK]": UNK_ID}
        for key in sorted({key for sequence in train["log_keys"] for key in sequence}):
            if key not in self.vocabulary:
                self.vocabulary[key] = len(self.vocabulary)
        self.max_events = max(map(len, train["log_keys"]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = LogBERTEncoder(len(self.vocabulary), self.max_events + 1,
                                      self.config["logbert_dim"]).to(device)
        normal_train = [self._map_keys(x) for x in train.loc[train["_target"].eq(0), "log_keys"]]
        normal_validation = [self._map_keys(x) for x in validation.loc[validation["_target"].eq(0), "log_keys"]]
        if not normal_train or not normal_validation:
            raise ValueError("LogBERT requires normal sessions in the training and validation partitions.")
        def loader(values, masked=True, shuffle=False):
            return DataLoader(MaskedSessions(values, masked), batch_size=self.config["batch_size"],
                              shuffle=shuffle, collate_fn=_collate, num_workers=0)
        train_loader = loader(normal_train, shuffle=True)
        validation_loader = loader(normal_validation)
        self.encoder.eval()
        centers = []
        with torch.no_grad():
            for values, _, padding in loader(normal_train, masked=False):
                _, hidden = self.encoder(values.to(device), padding.to(device))
                centers.append(hidden[:, 0].cpu())
        center = torch.cat(centers).mean(dim=0).to(device).detach()
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        def epoch(data_loader, training):
            # Compute one training or validation pass.
            self.encoder.train(training)
            total = 0.0
            for values, labels, padding in data_loader:
                values, labels, padding = values.to(device), labels.to(device), padding.to(device)
                with torch.set_grad_enabled(training):
                    logits, hidden = self.encoder(values, padding)
                    loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    loss = loss + 0.1 * ((hidden[:, 0] - center) ** 2).mean()
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                        optimizer.step()
                total += float(loss.item()) * len(values)
            return total / len(data_loader.dataset)

        # Retain the checkpoint with the lowest normal-validation loss.
        best_loss, best_state, stale = float("inf"), None, 0
        for number in range(1, self.config["logbert_epochs"] + 1):
            progress(12 + int(30 * (number - 1) / self.config["logbert_epochs"]),
                     f"Training LogBERT epoch {number}/{self.config['logbert_epochs']}")
            train_loss, validation_loss = epoch(train_loader, True), epoch(validation_loader, False)
            self.history.append({"epoch": number, "train_loss": train_loss, "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-4:
                best_loss, best_state, stale = validation_loss, copy.deepcopy(self.encoder.state_dict()), 0
            else:
                stale += 1
                if stale >= 3:
                    break
        self.encoder.load_state_dict(best_state)
        self.encoder.eval()
        self.encoder.to("cpu")
        key_tfidf = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False)
        key_tfidf.fit([" ".join(sequence) for sequence in train["log_keys"]])
        self.key_idf = dict(zip(key_tfidf.get_feature_names_out(), key_tfidf.idf_))
        self.unknown_weight = float(max(key_tfidf.idf_))

    def _session_embeddings(self, sequences: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        # Extract DIST vectors and IDF-weighted contextual event vectors.
        mean_rows, weighted_rows = [], []
        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            for start in range(0, len(sequences), self.config["batch_size"]):
                batch = sequences[start:start + self.config["batch_size"]]
                values = pad_sequence([torch.tensor([DIST_ID] + self._map_keys(keys), dtype=torch.long)
                                       for keys in batch], batch_first=True, padding_value=PAD_ID).to(device)
                _, hidden = self.encoder(values, values.eq(PAD_ID))
                hidden = hidden.cpu().numpy()
                mean_rows.extend(hidden[:, 0])
                for index, keys in enumerate(batch):
                    keys = keys[:self.max_events]
                    weights = [self.key_idf.get(key, self.unknown_weight) for key in keys]
                    weighted_rows.append(np.average(hidden[index, 1:len(keys) + 1], weights=weights, axis=0))
        return np.asarray(mean_rows, dtype=np.float32), np.asarray(weighted_rows, dtype=np.float32)


def _classifier(code: str, config: dict):
    # Hold classifier settings fixed across representations.
    if code == "lr":
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear",
                                   random_state=config["seed"])
    return RandomForestClassifier(n_estimators=config["rf_trees"], class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=config["seed"])


def _prediction_rows(units: pd.DataFrame, probability: np.ndarray, threshold: float) -> pd.DataFrame:
    # Keep record-level evidence for TP, FP, FN, and TN inspection.
    columns = [c for c in ("row_id", "unit_id", "log_text", "session_id", "known_anomaly_type", "event_count") if c in units]
    result = units[columns].copy().reset_index(drop=True)
    result["score"] = probability.astype(float)
    result["predicted"] = (probability >= threshold).astype(int)
    result["confidence"] = np.where(result["predicted"].eq(1), probability, 1 - probability)
    result["threshold"] = threshold
    if "_target" in units:
        # Missing ground truth never becomes a made-up normal label.
        actual = units["_target"].reset_index(drop=True)
        result["actual"] = pd.Series([int(value) if pd.notna(value) and value in (0, 1) else None
                                     for value in actual], dtype=object)
        result["outcome"] = [_outcome(a, p) if a is not None else None
                             for a, p in zip(result["actual"], result["predicted"])]
    else:
        result["actual"], result["outcome"] = None, None
    return result


def train_experiment(frame: pd.DataFrame, config: dict,
                     progress_callback: Callable[[int, str], None] | None = None) -> dict:
    """Train selected configurations and return all models plus held-out evidence."""
    # Validate units and freeze one shared train/validation/test partition.
    options = _settings(config)
    progress = progress_callback or (lambda percent, message: None)
    _seed(options["seed"])
    progress(3, "Validating fixed experimental partitions")
    units = _make_units(frame, options["group"], training=True)
    if "_session" in frame:
        # Blank identifiers represent ungrouped logs, not one shared session.
        known_sessions = frame["_session"].notna() & frame["_session"].astype(str).str.strip().ne("")
        if frame.loc[known_sessions].groupby("_session")["_split"].nunique().gt(1).any():
            raise ValueError("Related sessions must stay in one partition.")
    partitions = [units.loc[units["_split"].eq(name)].reset_index(drop=True)
                  for name in ("train", "validation", "test")]
    for name, part in zip(("train", "validation", "test"), partitions):
        if part.empty or set(part["_target"].unique()) != {0, 1}:
            raise ValueError(f"The {name} partition needs both normal and anomalous units.")
    train, validation, test = partitions
    targets = [part["_target"].to_numpy(dtype=int) for part in partitions]
    selected_reps = list(REPRESENTATIONS) if options["representation"] == "all" else [options["representation"]]
    selected_clfs = list(CLASSIFIERS) if options["classifier"] == "all" else [options["classifier"]]
    required_reps = [r for r in selected_reps if r != "fusion"]
    if "fusion" in selected_reps:
        required_reps = list(dict.fromkeys([*required_reps, "tfidf", "embedding"]))
    progress(8, "Fitting representations on the training partition")
    features = FeatureSpace(options, required_reps)
    matrices = [features.fit(train, validation, progress), features.transform(validation), features.transform(test)]
    progress(46, "Feature extraction complete; training classifiers")
    models, payloads, results, predictions = {}, {}, [], {}
    jobs = len(required_reps) * len(selected_clfs)
    done = 0

    def store_model(rep, clf, payload):
        # Publish every selected configuration and its test records.
        model_id = f"{options['group']}__{rep}__{clf}"
        models[model_id] = payload
        metrics = _metrics(targets[2], payload["test_probability"], payload["threshold"])
        result = {"model_id": model_id, "representation": representation_name(rep, options["group"]),
                  "representation_code": rep, "classifier": CLASSIFIERS[clf], "classifier_code": clf,
                  "group": options["group"], "unit": "log" if options["group"] == "logs" else "session",
                  "threshold": payload["threshold"], "validation_f1": payload["validation_f1"],
                  "train_seconds": payload["train_seconds"] + features.training_seconds,
                  "classifier_train_seconds": payload["train_seconds"],
                  "feature_train_seconds": features.training_seconds,
                  "predict_seconds": payload["predict_seconds"], **metrics}
        if rep == "fusion":
            result["fusion_weight_tfidf"] = payload["alpha"]
        results.append(result)
        predictions[model_id] = _prediction_rows(test, payload["test_probability"], payload["threshold"])

    # Train base estimators and tune their thresholds on validation only.
    for rep in required_reps:
        for clf in selected_clfs:
            progress(47 + int(43 * done / max(jobs, 1)),
                     f"Training {representation_name(rep, options['group'])} · {CLASSIFIERS[clf]}")
            estimator = _classifier(clf, options)
            start = time.perf_counter()
            estimator.fit(matrices[0][rep], targets[0])
            train_seconds = time.perf_counter() - start
            validation_probability = estimator.predict_proba(matrices[1][rep])[:, 1]
            threshold, validation_f1 = _threshold(targets[1], validation_probability)
            start = time.perf_counter()
            test_probability = estimator.predict_proba(matrices[2][rep])[:, 1]
            payload = {"representation": rep, "classifier": clf, "estimator": estimator,
                       "threshold": threshold, "validation_f1": validation_f1,
                       "validation_probability": validation_probability, "test_probability": test_probability,
                       "train_seconds": train_seconds, "predict_seconds": time.perf_counter() - start}
            payloads[(rep, clf)] = payload
            if rep in selected_reps:
                store_model(rep, clf, payload)
            done += 1

    # Search fusion weights on validation probabilities without touching test labels.
    if "fusion" in selected_reps:
        progress(92, "Selecting decision-fusion weights using validation results")
        for clf in selected_clfs:
            lexical, embedding = payloads[("tfidf", clf)], payloads[("embedding", clf)]
            best = {"validation_f1": -1.0}
            for alpha in np.linspace(0.0, 1.0, 21):
                values = alpha * lexical["validation_probability"] + (1 - alpha) * embedding["validation_probability"]
                threshold, validation_f1 = _threshold(targets[1], values)
                if validation_f1 > best["validation_f1"]:
                    best = {"alpha": float(alpha), "threshold": threshold, "validation_f1": validation_f1}
            payload = {**best, "representation": "fusion", "classifier": clf,
                       "estimators": {"tfidf": lexical["estimator"], "embedding": embedding["estimator"]},
                       "test_probability": best["alpha"] * lexical["test_probability"] +
                                           (1 - best["alpha"]) * embedding["test_probability"],
                       "train_seconds": lexical["train_seconds"] + embedding["train_seconds"],
                       "predict_seconds": lexical["predict_seconds"] + embedding["predict_seconds"]}
            store_model("fusion", clf, payload)

    # Report fitted dimensions and preserve all reusable inference artifacts.
    dimension = options["word2vec_dim"] if options["group"] == "logs" else options["logbert_dim"]
    vocabulary_size = len(features.tfidf.vocabulary_)
    feature_info = {"vocabulary_size": vocabulary_size, "embedding_dimension": dimension if features.requires_embedding else 0,
                    "vector_sizes": {"tfidf": vocabulary_size, "embedding": dimension,
                                     "concat": vocabulary_size + dimension, "weighted": dimension,
                                     "fusion": "two probability scores"},
                    "template_vocabulary_size": len(features.vocabulary), "max_session_events": features.max_events,
                    "train_units": len(train), "validation_units": len(validation), "test_units": len(test),
                    "min_document_frequency": 2, "ngram_range": [1, 2],
                    "embedding_method": "Word2Vec skip-gram" if options["group"] == "logs" else "Custom LogBERT-style masked-event Transformer",
                    "fitting_policy": "Features and scalers: training only; thresholds and fusion: validation only; metrics: test only"}
    progress(100, f"Completed {len(results)} configurations")
    return {"results": results, "predictions": predictions, "models": models, "features": features,
            "group": options["group"], "config": options, "training_history": features.history,
            "feature_info": feature_info}


def _model_probability(payload: dict, matrices: dict) -> np.ndarray:
    # Use the same estimators and fusion weight used during evaluation.
    if payload["representation"] == "fusion":
        alpha = payload["alpha"]
        return (alpha * payload["estimators"]["tfidf"].predict_proba(matrices["tfidf"])[:, 1]
                + (1 - alpha) * payload["estimators"]["embedding"].predict_proba(matrices["embedding"])[:, 1])
    return payload["estimator"].predict_proba(matrices[payload["representation"]])[:, 1]


def predict_bundle(bundle: dict, frame_or_text: pd.DataFrame | str | list[str], *, session: bool = False) -> list[dict]:
    """Predict all saved configurations; strings must already use the project's cleaner."""
    # Session predictions require complete normalized session records.
    group = bundle["group"]
    if group == "sessions" and (not session or not isinstance(frame_or_text, pd.DataFrame)):
        raise ValueError("Session models require a complete session DataFrame and session=True.")
    if group == "logs" and session:
        raise ValueError("This run evaluates individual log entries, not sessions.")
    if isinstance(frame_or_text, pd.DataFrame):
        frame = frame_or_text
    else:
        texts = [frame_or_text] if isinstance(frame_or_text, str) else list(frame_or_text)
        frame = pd.DataFrame({"_text": texts, "log_text": texts})
    units = _make_units(frame, group, training=False)
    matrices = bundle["features"].transform(units)
    output = []
    for model_id, payload in bundle["models"].items():
        probability = _model_probability(payload, matrices)
        rows = _prediction_rows(units, probability, payload["threshold"])
        for row in rows.to_dict("records"):
            row.update({"model_id": model_id, "representation": representation_name(payload["representation"], group),
                        "classifier": CLASSIFIERS[payload["classifier"]], "group": group,
                        "unit": "log" if group == "logs" else "session",
                        "prediction": "anomaly" if row["predicted"] else "normal",
                        "score_note": "Model probability score; confidence has not been separately calibrated."})
            if group == "sessions":
                keys = units.loc[units["unit_id"].eq(row["unit_id"]), "log_keys"].iloc[0]
                row["truncated_events"] = max(0, len(keys) - bundle["features"].max_events) if bundle["features"].requires_embedding else 0
                row["unseen_events"] = sum(key not in bundle["features"].vocabulary for key in keys) if bundle["features"].requires_embedding else 0
            output.append(row)
    return output


def explain_prediction(bundle: dict, model_id: str, text: str, max_terms: int = 6) -> dict:
    """Explain a cleaned log using model coefficients or token-removal sensitivity."""
    # Never label session features as individual-token explanations.
    if model_id not in bundle["models"]:
        raise ValueError("This configuration is not present in the selected run.")
    if bundle["group"] == "sessions":
        return {"method": "session_context", "terms": [],
                "note": "This configuration classifies the whole ordered session. A single log's tokens do not explain that session decision; LogBERT methods encode the sequence of event-template identifiers."}
    payload, features = bundle["models"][model_id], bundle["features"]
    if payload["representation"] == "tfidf" and payload["classifier"] == "lr":
        vector = features.tfidf.transform([text]).tocsr()
        names = features.tfidf.get_feature_names_out()
        contributions = vector.data * payload["estimator"].coef_[0, vector.indices]
        evidence = sorted(zip(names[vector.indices], contributions), key=lambda item: -abs(item[1]))[:max_terms]
        return {"method": "linear_contributions", "terms": [{"term": str(term), "contribution": float(value)} for term, value in evidence],
                "note": "Signed TF-IDF × coefficient contributions to anomaly log-odds; positive supports anomaly, negative supports normal."}
    tokens = str(text).split()
    unique = list(dict.fromkeys(tokens))[:24]
    if not unique:
        return {"method": "token_removal", "terms": [], "note": "No tokens are available to explain."}
    alternatives = [text] + [" ".join(token for token in tokens if token != removed) for removed in unique]
    units = _make_units(pd.DataFrame({"_text": alternatives}), "logs", training=False)
    probabilities = _model_probability(payload, features.transform(units))
    evidence = sorted(zip(unique, probabilities[0] - probabilities[1:]), key=lambda item: -abs(item[1]))[:max_terms]
    return {"method": "token_removal", "terms": [{"term": term, "contribution": float(value)} for term, value in evidence],
            "note": "Change in anomaly score when every occurrence of a token is removed. This is local model sensitivity, not a causal explanation."}
