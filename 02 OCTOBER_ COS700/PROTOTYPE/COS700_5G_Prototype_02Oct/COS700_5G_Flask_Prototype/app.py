"""Local Flask application for the COS700 controlled NLP experiments."""

# Keep application state and trained artifacts within this project.
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from data import apply_split, clean_log_text, parse_file, prepare_data

ROOT = Path(__file__).resolve().parent
INSTANCE = Path(os.environ.get("COS700_INSTANCE", ROOT / "instance_v2")).resolve()
for folder in ("datasets", "runs", "analyses"):
    (INSTANCE / folder).mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=64 * 1024 * 1024, JSON_SORT_KEYS=False)
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cos700-training")
DATASETS, BUNDLES, JOBS = {}, {}, {}
ACTIVE_DATASET = None
ACTIVE_JOB = None


def safe(value):
    # Convert scientific values to valid JSON, including missing values.
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [safe(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reply(value, code=200):
    # Use one serialization policy for every endpoint.
    return jsonify(safe(value)), code


def now():
    # Timestamp saved experiments in UTC.
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    # Publish complete metadata atomically.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def identifier(value):
    # Restrict storage identifiers to application-generated names.
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError("Invalid dataset or run identifier.")
    return value


def register_dataset(raw, name, *, default=False, dataset_id=None):
    # Normalize inputs once and preserve the prepared dataset for saved runs.
    frame, context = prepare_data(raw, source_name=name, is_default=default)
    dataset_id = dataset_id or uuid.uuid4().hex[:16]
    context.update(id=dataset_id, created_at=now())
    frame.to_pickle(INSTANCE / "datasets" / f"{dataset_id}.pkl")
    write_json(INSTANCE / "datasets" / f"{dataset_id}.json", context)
    DATASETS[dataset_id] = (frame, context)
    return frame, context


def load_dataset(dataset_id=None):
    # Only deserialize datasets created by this application.
    key = dataset_id or ACTIVE_DATASET
    if not key:
        raise ValueError("No dataset has been uploaded. Upload a CSV file first.")
    key = identifier(key)
    if key not in DATASETS:
        metadata = INSTANCE / "datasets" / f"{key}.json"
        if not metadata.exists():
            raise ValueError("That dataset is not available.")
        DATASETS[key] = (pd.read_pickle(metadata.with_suffix(".pkl")), json.loads(metadata.read_text(encoding="utf-8")))
    return DATASETS[key]


def initialize():
    # Start with no active dataset. The user must upload a CSV before the workspace has data.
    global ACTIVE_DATASET
    ACTIVE_DATASET = None


def run_metadata(run_id):
    # Keep model loading separate from inexpensive results browsing.
    path = INSTANCE / "runs" / identifier(run_id) / "run.json"
    if not path.exists():
        raise ValueError("The requested run is not available.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_bundle(run_id):
    # Keep only a small number of trusted trained bundles in memory.
    run_id = identifier(run_id)
    with LOCK:
        if run_id not in BUNDLES:
            run_metadata(run_id)
            if len(BUNDLES) >= 2:
                BUNDLES.pop(next(iter(BUNDLES)))
            BUNDLES[run_id] = joblib.load(INSTANCE / "runs" / run_id / "models.joblib")
        return BUNDLES[run_id]


def run_list():
    # Present the most recently completed real experiments first.
    result = []
    for path in (INSTANCE / "runs").glob("*/run.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        result.append({k: item.get(k) for k in ("id", "status", "created_at", "config", "dataset_id", "dataset_name", "recommended_model_id")})
    return sorted(result, key=lambda item: item["created_at"], reverse=True)


def new_job(kind, **details):
    # Serialize expensive work to protect memory and deterministic seeds.
    global ACTIVE_JOB
    with LOCK:
        if ACTIVE_JOB and JOBS.get(ACTIVE_JOB, {}).get("status") in {"queued", "running"}:
            raise ValueError("A job is already running. Wait for it to finish before starting another.")
        job = dict(id=uuid.uuid4().hex[:16], kind=kind, status="queued", progress=0,
                   message="Waiting to start", created_at=now(), **details)
        JOBS[job["id"]] = job
        ACTIVE_JOB = job["id"]
        return job


def update_job(job, **values):
    # Publish background progress without exposing partial results.
    with LOCK:
        job.update(values)


def training_worker(job, frame, context, config):
    # Run actual experiments and retain all fitted configurations.
    try:
        from engine import train_experiment
        update_job(job, status="running", message="Checking dataset partitions")
        prepared = apply_split(frame, mode=config["split_mode"], test_size=config["test_size"],
                               validation_size=config["validation_size"], seed=config["seed"])
        bundle = train_experiment(prepared, config,
                                  lambda percent, message: update_job(job, progress=min(percent, 98), message=message))
        run_id = uuid.uuid4().hex[:16]
        directory = INSTANCE / "runs" / run_id
        directory.mkdir()
        best = max(bundle["results"], key=lambda result: result["validation_f1"])
        metadata = dict(id=run_id, status="completed", created_at=now(), dataset_id=context["id"],
                        dataset_name=context["name"], config=config, results=bundle["results"],
                        feature_info=bundle["feature_info"], training_history=bundle["training_history"],
                        recommended_model_id=best["model_id"], selection_basis="Highest validation F1",
                        actual_split_counts=prepared["_split"].value_counts().to_dict())
        joblib.dump(bundle, directory / "models.joblib", compress=3)
        for model_id, evidence in bundle["predictions"].items():
            evidence.to_csv(directory / f"{model_id}.csv", index=False)
        pd.DataFrame(bundle["results"]).drop(columns=["confusion_matrix"], errors="ignore").to_csv(directory / "metrics.csv", index=False)
        prepared[["row_id", "_split"]].to_csv(directory / "partitions.csv", index=False)
        write_json(directory / "run.json", metadata)
        with LOCK:
            if len(BUNDLES) >= 2:
                BUNDLES.pop(next(iter(BUNDLES)))
            BUNDLES[run_id] = bundle
        update_job(job, status="completed", progress=100, message="Training and held-out evaluation complete", run_id=run_id)
    except Exception as exc:
        app.logger.error("Training failed\n%s", traceback.format_exc())
        update_job(job, status="failed", error=str(exc), message=str(exc))


def page_values(frame, *, maximum=None):
    # Page only the displayed records without changing the source dataset.
    size = max(1, min(100, int(request.args.get("page_size", 20))))
    total = len(frame)
    available = min(total, maximum) if maximum is not None else total
    pages = max(1, math.ceil(available / size))
    page = max(1, min(pages, int(request.args.get("page", 1))))
    visible = frame.iloc[:available].iloc[(page - 1) * size:page * size]
    return dict(rows=visible.to_dict("records"), total=total, available=available, page=page, pages=pages, page_size=size)


def outcome_summary(rows, frame, group):
    # Derive counts and timeline points from actual predicted outputs.
    summary = {"normal": int(rows["predicted"].eq(0).sum()), "anomaly": int(rows["predicted"].eq(1).sum())}
    if group == "sessions":
        times = frame.groupby("_session")["_timestamp"].min()
        mapped = rows["session_id"].astype(str).map(times)
    else:
        times = frame.set_index(frame["row_id"].astype(str))["_timestamp"]
        mapped = rows["row_id"].astype(str).map(times)
    hourly = pd.to_datetime(mapped, utc=True, errors="coerce").dt.floor("h")
    counts = pd.DataFrame({"time": hourly, "anomaly": rows["predicted"].to_numpy()}).dropna().groupby("time")["anomaly"].sum()
    timeline = [{"time": time.isoformat(), "count": int(count)} for time, count in counts.items()]
    return summary, timeline


def prediction_frame(run_id, model_id=None):
    # Read persisted per-record evidence without rerunning the model.
    meta = run_metadata(run_id)
    model_id = model_id or meta["recommended_model_id"]
    if model_id not in {r["model_id"] for r in meta["results"]}:
        raise ValueError("Choose a configuration from this run.")
    rows = pd.read_csv(INSTANCE / "runs" / run_id / f"{model_id}.csv", keep_default_na=False, float_precision="round_trip",
                       dtype={"row_id": str, "unit_id": str, "session_id": str})
    return rows, meta, model_id


def english_explanation(bundle, model_id, clean_text, predicted, score, threshold):
    """Turn the model evidence into a short English explanation for a table row."""
    from engine import explain_prediction
    evidence = explain_prediction(bundle, model_id, str(clean_text))
    direction = "anomalous" if int(predicted) == 1 else "normal"
    relation = "above" if int(predicted) == 1 else "below"
    base = f"The log was classified as {direction} because its anomaly score ({float(score):.4f}) is {relation} the model decision threshold ({float(threshold):.4f})."
    terms = evidence.get("terms") or []
    if terms:
        ranked = sorted(terms, key=lambda item: abs(float(item.get("contribution", 0))), reverse=True)[:4]
        signals = ", ".join(f"{item.get('term')} ({float(item.get('contribution', 0)):+.4f})" for item in ranked)
        base += f" The strongest local model signals were {signals}."
    note = evidence.get("note")
    if note and evidence.get("method") == "session_context":
        base += " The selected session model evaluates the ordered event-template sequence as a whole, so individual words are not treated as causal evidence."
    return base


def analysis_row_enrichment(rows, frame, bundle, model_id):
    """Add the display fields and English explanations required by result tables."""
    result = rows.copy()
    if bundle["group"] == "sessions":
        units = frame.copy()
        units["_session"] = units["_session"].astype(str)
        cleaned = units.groupby("_session", sort=False)["_text"].apply(lambda values: " [EVENT] ".join(values.astype(str)))
        known = units.groupby("_session", sort=False)["_target"].first() if "_target" in units else pd.Series(dtype=object)
        result["cleaned_message"] = result["unit_id"].astype(str).map(cleaned).fillna(result.get("log_text", ""))
        result["known_label"] = result["unit_id"].astype(str).map(known).map(lambda value: int(value) if pd.notna(value) else None)
    else:
        lookup = frame.set_index(frame["row_id"].astype(str))
        result["cleaned_message"] = result["row_id"].astype(str).map(lookup["_text"]).fillna(result.get("log_text", ""))
        if "_target" in lookup:
            result["known_label"] = result["row_id"].astype(str).map(lookup["_target"]).map(lambda value: int(value) if pd.notna(value) else None)
        else:
            result["known_label"] = None
    result["explainability"] = [
        english_explanation(bundle, model_id, clean, pred, score, threshold)
        for clean, pred, score, threshold in zip(result["cleaned_message"], result["predicted"], result["score"], result["threshold"])
    ]
    return result


def analysis_worker(job, run_id, model_id, frame, context):
    # Apply one retained configuration to all records in the selected input.
    try:
        from engine import predict_bundle
        update_job(job, status="running", progress=10, message="Loading the trained configuration")
        bundle = load_bundle(run_id)
        meta = run_metadata(run_id)
        model_id = model_id or meta["recommended_model_id"]
        if model_id not in bundle["models"]:
            raise ValueError("Choose a configuration from this run.")
        if bundle["group"] == "sessions" and not context["can_sessions"]:
            raise ValueError("Session models require complete session_id and template_id values.")
        selected = {**bundle, "models": {model_id: bundle["models"][model_id]}}
        update_job(job, progress=30, message="Analysing the full dataset")
        inputs = frame.copy()
        if not context["has_labels"]:
            inputs = inputs.drop(columns="_target", errors="ignore")
        rows = pd.DataFrame(predict_bundle(selected, inputs, session=bundle["group"] == "sessions"))
        rows = rows[rows["model_id"].eq(model_id)].copy()
        rows = analysis_row_enrichment(rows, frame, bundle, model_id)
        summary, timeline = outcome_summary(rows, frame, bundle["group"])
        analysis_id = uuid.uuid4().hex[:16]
        rows.to_csv(INSTANCE / "analyses" / f"{analysis_id}.csv", index=False)
        info = dict(id=analysis_id, run_id=run_id, model_id=model_id, dataset_id=context["id"],
                    dataset_name=context["name"], created_at=now(), summary=summary, timeline=timeline,
                    unit="session" if bundle["group"] == "sessions" else "log", total=len(rows),
                    model=next(r for r in meta["results"] if r["model_id"] == model_id))
        write_json(INSTANCE / "analyses" / f"{analysis_id}.json", info)
        update_job(job, status="completed", progress=100, message="Dataset analysis complete", run_id=run_id, analysis_id=analysis_id)
    except Exception as exc:
        app.logger.error("Analysis failed\n%s", traceback.format_exc())
        update_job(job, status="failed", message=str(exc), error=str(exc))


@app.before_request
def local_request():
    # Block cross-site writes to the local research application.
    if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return reply({"error": "Cross-site requests are not permitted."}, 403)


@app.after_request
def response_headers(response):
    # Keep project data out of browser caches and unrelated frames.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(Exception)
def api_error(error):
    # Return readable errors while retaining detailed server logs.
    if isinstance(error, HTTPException):
        return reply({"error": error.description}, error.code)
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return reply({"error": str(error)}, 400)
    app.logger.error("Application error\n%s", traceback.format_exc())
    # This is a local research prototype. Return the concrete exception so a
    # model-artifact problem is actionable instead of hiding it behind a generic 500.
    return reply({"error": f"{type(error).__name__}: {error}"}, 500)


@app.get("/")
def index():
    # Serve the requested HTML interface.
    return render_template("index.html")


@app.get("/api/health")
def health():
    # Let the launcher recognize this application on the local port.
    return reply({"application": "COS700_5G_Flask_Prototype", "version": "bright-cards-v2", "status": "ready"})


@app.get("/api/state")
def state():
    # Return a blank workspace until a user-uploaded dataset becomes active.
    if not ACTIVE_DATASET:
        return reply(dict(dataset=None, runs=[], active_job=JOBS.get(ACTIVE_JOB), latest_analysis=None))
    _, context = load_dataset()
    analyses = sorted((INSTANCE / "analyses").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest = None
    for path in analyses:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("dataset_id") == context["id"]:
            latest = candidate
            break
    runs = [run for run in run_list() if run.get("dataset_id") == context["id"]]
    return reply(dict(dataset=context, runs=runs, active_job=JOBS.get(ACTIVE_JOB), latest_analysis=latest))


@app.post("/api/upload")
def upload():
    # Accept only supported data formats, never executable model files.
    global ACTIVE_DATASET
    file = request.files.get("file")
    if file is None or not file.filename:
        raise ValueError("Choose a CSV file.")
    name = secure_filename(file.filename) or "uploaded.csv"
    if Path(name).suffix.lower() != ".csv":
        raise ValueError("Choose a CSV file.")
    raw = parse_file(file.stream, name)
    if len(raw) > 250000:
        raise ValueError("This local prototype supports up to 250,000 input records per upload.")
    _, context = register_dataset(raw, name)
    ACTIVE_DATASET = context["id"]
    return reply(context)


@app.get("/api/logs")
def logs():
    # Filter and paginate records without sampling model training data.
    frame, _ = load_dataset(request.args.get("dataset_id"))
    if request.args.get("split"):
        frame = frame[frame["_split"].eq(request.args["split"])]
    if request.args.get("search"):
        frame = frame[frame["log_text"].astype(str).str.contains(request.args["search"], case=False, regex=False)]
    limit = request.args.get("limit", "100")
    if limit not in {"100", "1000", "all"}:
        raise ValueError("Display All, Top 100 or Top 1,000 logs.")
    public = frame[[c for c in frame.columns if not c.startswith("_")]]
    return reply(page_values(public, maximum=None if limit == "all" else int(limit)))


@app.get("/api/record/<int:row_id>")
def record(row_id):
    # Expose the original text, cleaned form and actual tokens.
    frame, _ = load_dataset(request.args.get("dataset_id"))
    found = frame[frame["row_id"].eq(row_id)]
    if found.empty:
        raise ValueError("That record does not exist in the selected dataset.")
    row = found.iloc[0]
    sid = str(row.get("_session", ""))
    return reply(dict(record={k: v for k, v in row.items() if not k.startswith("_")}, clean_text=row["_text"],
                      tokens=str(row["_text"]).split(), session_count=int(frame["_session"].eq(sid).sum()) if sid else 0))


@app.post("/api/train")
def train():
    # Resolve the controlled profile before dispatching background training.
    params = request.get_json(force=False) or {}
    frame, context = load_dataset(params.get("dataset_id"))
    if not context["has_labels"]:
        raise ValueError("Training requires normal/anomalous labels on every record.")
    group = params.get("group", "logs")
    if group not in {"logs", "sessions"}:
        raise ValueError("Choose a valid experiment group.")
    if group == "sessions" and not context["can_sessions"]:
        raise ValueError("Session experiments require real session and template IDs with consistent session labels.")
    profile = params.get("profile", "quick")
    if profile not in {"quick", "research"}:
        raise ValueError("Choose the quick or research profile.")
    config = dict(group=group, representation=params.get("representation", "all"),
                  classifier=params.get("classifier", "all"), split_mode=params.get("split_mode", "provided"),
                  test_size=float(params.get("test_size", .15)), validation_size=float(params.get("validation_size", .15)),
                  profile=profile, seed=42, rf_trees=120 if profile == "quick" else 300,
                  w2v_epochs=5 if profile == "quick" else 10, logbert_epochs=5 if profile == "quick" else 12,
                  word2vec_dim=100, logbert_dim=64, max_features=20000 if group == "logs" else 25000,
                  batch_size=256, dataset_id=context["id"])
    if config["representation"] not in {"all", "tfidf", "embedding", "concat", "weighted", "fusion"} or config["classifier"] not in {"all", "lr", "rf"}:
        raise ValueError("Choose valid representations and classifiers.")
    apply_split(frame, mode=config["split_mode"], test_size=config["test_size"], validation_size=config["validation_size"], seed=42)
    job = new_job("training", dataset_id=context["id"])
    POOL.submit(training_worker, job, frame.copy(), context.copy(), config)
    return reply({"job_id": job["id"]}, 202)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    # Report background job progress for the browser.
    if job_id not in JOBS:
        raise ValueError("This job is no longer active. Completed results remain in Saved Runs.")
    return reply(JOBS[job_id])


@app.get("/api/runs/<run_id>")
def run(run_id):
    # Serve measured results from a completed experiment.
    return reply(run_metadata(run_id))


@app.get("/api/runs/<run_id>/predictions")
def predictions(run_id):
    # Document actual held-out TP, FP, FN and TN evidence.
    rows, meta, model_id = prediction_frame(run_id, request.args.get("model_id"))
    frame, _ = load_dataset(meta["dataset_id"])
    summary, timeline = outcome_summary(rows, frame, meta["config"]["group"])
    outcome = request.args.get("outcome", "all")
    if outcome not in {"all", "TP", "FP", "FN", "TN"}:
        raise ValueError("Unknown prediction outcome.")
    visible = rows if outcome == "all" else rows[rows["outcome"].eq(outcome)]
    return reply({**page_values(visible), "summary": summary, "timeline": timeline, "model_id": model_id})


@app.post("/api/predict")
def predict():
    # Score new input with retained artifacts from the selected run.
    from engine import explain_prediction, predict_bundle
    params = request.get_json() or {}
    bundle = load_bundle(params.get("run_id"))
    frame, context = load_dataset(params.get("dataset_id"))
    session = bundle["group"] == "sessions"
    row_id = params.get("row_id")
    if row_id not in (None, ""):
        found = frame[frame["row_id"].eq(int(row_id))]
        if found.empty:
            raise ValueError("Select an existing dataset record.")
        selected = found.iloc[0]
        if session:
            if not context["can_sessions"]:
                raise ValueError("The selected input has no complete session/template context.")
            inputs = frame[frame["_session"].eq(selected["_session"])].copy()
        else:
            inputs = found.copy()
        cleaned = str(selected["_text"])
        context_text = f"Session {selected['_session']} · {len(inputs)} ordered logs" if session else f"Dataset record {row_id}"
    else:
        text = str(params.get("text", "")).strip()
        if not text or len(text) > 50000:
            raise ValueError("Enter a non-empty log message of at most 50,000 characters.")
        if session:
            raise ValueError("This LogBERT group needs an ordered session with template IDs. Select a dataset record to retrieve its session.")
        cleaned, context_text = clean_log_text(text), "Pasted log message"
        inputs = cleaned
    if isinstance(inputs, pd.DataFrame) and not context["has_labels"]:
        inputs = inputs.drop(columns="_target", errors="ignore")
    output = predict_bundle(bundle, inputs, session=session)
    model_id = params.get("model_id") or run_metadata(params["run_id"])["recommended_model_id"]
    explanation = explain_prediction(bundle, model_id, cleaned)
    explanation["model_id"] = model_id
    return reply(dict(predictions=output, context=context_text, explanation=explanation,
                      clean_text=cleaned, score_note="Anomaly probability and threshold-selected confidence are model estimates, not calibrated certainty."))


@app.post("/api/manual-analyze")
def manual_analyze():
    # Analyse exactly the user-entered sample logs with the recommended model from the latest run.
    from engine import predict_bundle
    params = request.get_json() or {}
    texts = params.get("logs")
    if not isinstance(texts, list) or not texts:
        raise ValueError("Enter at least one sample log.")
    if len(texts) > 100:
        raise ValueError("Enter no more than 100 sample logs at a time.")
    cleaned = [clean_log_text(str(value).strip()) for value in texts]
    if any(not value for value in cleaned):
        raise ValueError("Every sample log must contain text.")
    frame, context = load_dataset(params.get("dataset_id"))
    runs = [run for run in run_list() if run.get("dataset_id") == context["id"] and run.get("status") == "completed"]
    if not runs:
        raise ValueError("Train a model before analysing manual logs.")
    latest = runs[0]
    latest_meta = run_metadata(latest["id"])
    model_id = latest_meta.get("recommended_model_id")
    bundle = load_bundle(latest["id"])
    if not model_id or model_id not in bundle.get("models", {}):
        raise ValueError("The selected trained run does not contain its recommended model artifact. Please train the model again in Tab 3.")
    session = bundle["group"] == "sessions"
    if session:
        records = []
        for index, text in enumerate(cleaned, start=1):
            records.append({"row_id": str(index), "unit_id": str(index), "session_id": f"manual-{index}",
                            "_session": f"manual-{index}", "_template": "MANUAL-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                            "_text": text, "log_text": text, "event_count": 1})
        inputs = pd.DataFrame(records)
    else:
        inputs = cleaned
    # Pass only the selected best model to inference so the manual-analysis result
    # cannot be confused with another configuration from the same training run.
    prediction_rows = predict_bundle({**bundle, "models": {model_id: bundle["models"][model_id]}}, inputs, session=session)
    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise ValueError("The best trained model returned no predictions for the entered logs. Please train the model again in Tab 3.")
    predictions = predictions[predictions["model_id"].eq(model_id)].copy()
    predictions["record"] = [f"Manual log {i}" for i in range(1, len(predictions) + 1)]
    predictions["cleaned_message"] = cleaned[:len(predictions)]
    predictions["known_label"] = None
    predictions["explainability"] = [
        english_explanation(bundle, model_id, clean, pred, score, threshold)
        for clean, pred, score, threshold in zip(predictions["cleaned_message"], predictions["predicted"], predictions["score"], predictions["threshold"])
    ]
    model_result = next((item for item in latest_meta.get("results", []) if item.get("model_id") == model_id), None)
    if model_result is None:
        raise ValueError("The trained run is missing the result metadata for its recommended model. Please train the model again in Tab 3.")
    return reply({"run_id": latest["id"], "model_id": model_id, "model": model_result,
                  "rows": predictions[["record", "cleaned_message", "predicted", "score", "known_label", "explainability"]].to_dict("records"),
                  "total": len(predictions), "summary": {"normal": int(predictions["predicted"].eq(0).sum()), "anomaly": int(predictions["predicted"].eq(1).sum())}})


@app.post("/api/analyze")
def analyze():
    # Analyse a labelled or unlabelled upload with a saved model.
    params = request.get_json() or {}
    meta = run_metadata(params.get("run_id"))
    frame, context = load_dataset(params.get("dataset_id"))
    job = new_job("analysis", dataset_id=context["id"])
    POOL.submit(analysis_worker, job, meta["id"], params.get("model_id"), frame.copy(), context.copy())
    return reply({"job_id": job["id"]}, 202)


@app.get("/api/analyses/<analysis_id>")
def analysis(analysis_id):
    # Page completed batch predictions and show their actual summary.
    path = INSTANCE / "analyses" / f"{identifier(analysis_id)}.json"
    if not path.exists():
        raise ValueError("That analysis is not available.")
    info = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.read_csv(path.with_suffix(".csv"), keep_default_na=False, float_precision="round_trip")
    return reply({**info, **page_values(rows)})


@app.get("/api/export/<run_id>")
def export(run_id):
    # Export measured metrics or full per-record evidence as CSV.
    meta = run_metadata(run_id)
    kind = request.args.get("kind", "metrics")
    if kind == "metrics":
        path = INSTANCE / "runs" / run_id / "metrics.csv"
    elif kind == "predictions":
        _, _, model_id = prediction_frame(run_id, request.args.get("model_id"))
        path = INSTANCE / "runs" / run_id / f"{model_id}.csv"
    else:
        raise ValueError("Export metrics or predictions.")
    return send_file(path, as_attachment=True, download_name=f"{run_id}_{path.name}")


@app.get("/api/saved-results")
def saved_results():
    # Label supplied notebook outputs as historical results, not new training.
    return reply(json.loads((ROOT / "data" / "notebook_results.json").read_text(encoding="utf-8")))


# Initialize an empty workspace and start a local-only server.
initialize()
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=int(os.environ.get("PORT", "5081")), threads=6)
