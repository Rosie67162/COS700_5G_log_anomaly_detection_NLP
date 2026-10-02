# COS700 5G Network Log Analysis

A local research application built with Python, Flask, HTML, CSS and JavaScript. The interface uses the approved dark design with bright accents and the **Get started** panel.

## Open the prototype

On this computer, double-click **Start Prototype.cmd**. The application opens at **http://127.0.0.1:5055**. It runs locally; uploaded logs remain in this project. Keep the extracted project folder intact.

On another computer, install **Python 3.12** first. The launcher creates a project environment and downloads the packages on its first run. CPU PyTorch is installed separately from the official CPU package index. Initial setup needs internet access; ordinary use after installation does not.

For a manual start, install `requirements.txt`, install `torch==2.14.0` using `https://download.pytorch.org/whl/cpu`, then run `python app.py`. Windows paths with spaces must be quoted. The default port is 5055; the `PORT` environment variable can override it for a manual start.

## Use the application

1. **Logs & preprocessing:** upload the required CSV, browse the records, and inspect original text, cleaned text and tokens. The prototype starts blank until a CSV is loaded.
2. **Data context:** inspect dataset log entries, labelled normal/anomalous counts, label balance, partitions, metadata, audit information and anomaly counts over time.
3. **Train models:** choose log-level TF-IDF/Word2Vec or session-level TF-IDF/LogBERT-style experiments, an individual configuration or all configurations, and a training profile. Training runs in the background. The display limit never reduces training data.
4. **Evaluate & test:** review five metrics, the confusion matrix, and individual TP/FP/FN/TN records. CSV exports contain all held-out predictions, not only the visible page.
5. **Compare methods:** compare configurations from one run under identical conditions. The original notebook results are clearly labelled as historical reference results.
6. **Manual log analysis:** choose how many sample logs to enter, create exactly that many input boxes, then press **Analyse logs**. The latest trained run's recommended model is used and each result can be expanded to show English explainability.
7. **Dashboard:** after a model has been trained, view the best-model accuracy, metrics, confusion matrix, predicted normal/anomalous counts and the full dataset analysis table. Dataset analysis is started automatically with the best model. Click a result row to expand its English explainability.

## Research method

The prototype starts with no active dataset. Upload a CSV from the Logs & preprocessing tab before any dataset context, training, dashboard results or predictions are available. The app does not ship with an active/default dataset.

The cleaner preserves severity words, negation and 5G terms; replaces variable identifiers and numeric values; and retains original messages. Stop-word removal and stemming are not applied. If `clean_text` is supplied, that prepared text is retained. Only exact duplicate source rows are removed; recurring cleaned messages are audited rather than silently discarded.

When a supplied CSV contains partitions, those partitions are used by default. Custom partitions preserve linked scenario/session groups, so percentages can be approximate when group sizes vary. Every representation in one run uses the same partition and seed 42. Features and scalers fit on training data only; thresholds and fusion weights use validation F1. The recommended configuration is selected using validation F1, not test performance.

Each group has five representations and two classifiers: TF-IDF, embeddings, concatenation, TF-IDF-weighted embeddings and decision fusion; Logistic Regression and Random Forest. This gives ten configurations per group and twenty across both groups. The session encoder is a custom **LogBERT-style** masked-event Transformer trained on normal training sessions. It is not a downloaded pretrained LogBERT model.

| Profile | Random Forest trees | Word2Vec epochs | Maximum LogBERT epochs | Dataset |
|---|---:|---:|---:|---|
| Quick | 120 | 5 | 5 | All records |
| Research | 300 | 10 | 12 | All records |

Embedding dimensions are 100 (Word2Vec) and 64 (LogBERT-style). TF-IDF caps are 20,000 (logs) and 25,000 (sessions). Early stopping can shorten LogBERT training. Small numerical differences from Colab are possible across CPU/GPU, library versions and execution order. Timings are displayed separately for shared features and classifiers; they are not a live network throughput guarantee.

The session model uses ordered template IDs, not arbitrary isolated pasted text. Unknown templates and truncated events are returned in prediction metadata. Known anomaly categories are dataset annotations; these binary classifiers do not predict root causes or multiclass categories. TF-IDF/LR explanations show signed feature contributions; other supported log explanations show token-removal sensitivity, not causal explanations.

## Upload formats

The application upload control requires a CSV file. CSV data needs `log_text`, `message`, `text`, `log_message`, `log`, or `clean_text`. Binary `target` (0/1) or `label` (normal/abnormal) is required for training, but not for prediction. Real `session_id` and `template_id` fields are required for session experiments. An optional `split` column accepts train, validation and test. Timestamp and anomaly-category metadata are optional.

Uploads are limited to 64 MB and 250,000 records. Incorrect labels, incompatible partitions and incomplete session context produce explicit messages. Unlabelled uploads receive predictions, not fabricated accuracy metrics.

## Saved work and verification

`instance/runs` contains measured results, partition assignments, every fitted configuration and full test prediction CSVs. `instance/analyses` contains batch predictions. `instance/datasets` contains application-created dataset copies. Only local artifacts created by the app are deserialized; uploaded pickle/model files are not supported.

`TEST_RESULTS.md` documents the real full-dataset verification, model outcomes and any limitations. Existing notebook result tables are stored separately in `data/notebook_results.json`.

After installing the dependencies, run `python -m unittest discover -s tests -v` from the project folder. The saved-run verification is skipped when no bundled runs are present because this delivery intentionally starts without a dataset or trained artifacts.

All logical sections of the Python, HTML, CSS and JavaScript source have short explanatory comments. The prototype source is modified in this delivery to implement the requested workflow changes.

## Updated prototype workflow

The current interface uses this tab order:
1. Logs & preprocessing
2. Data context
3. Train models
4. Evaluate & test
5. Compare methods
6. Manual log analysis
7. Dashboard

A fresh launch starts without an active dataset. Upload a CSV in Logs & preprocessing before the other data-dependent pages populate. The Dashboard remains empty until a model has been trained for the uploaded dataset. The launcher uses local port 5067 so an older 5055 instance cannot be reused accidentally.
