# Prototype validation results

## Current delivery structural checks (2026-09-30)

- Tab navigation order is exactly: Logs & preprocessing, Data context, Train models, Evaluate & test, Compare methods, Manual log analysis, Dashboard.
- The Dashboard section is the final page section and is not the initial page.
- Dashboard content is hidden unless both an uploaded dataset and a completed trained model exist.
- Data Context content is hidden until a dataset is uploaded.
- Logs & preprocessing accepts CSV only and contains no dataset-analysis controls.
- The old Try a log page and all Try-a-log/default-dataset controls are removed.
- Manual log analysis creates exactly the requested number of input boxes and sends all entered logs to the best trained model when Analyse logs is pressed.
- Dashboard dataset analysis is automatically started for the latest run's recommended model and its result table contains Record/session, cleaned message, prediction, anomaly score and known label, with expandable English explainability.
- JavaScript syntax check passed with `node --check static/app.js`.
- Python syntax checks passed with `python -m py_compile app.py data.py engine.py`.
- All JavaScript element references used by `$()` resolve to an element ID in the HTML.
- The delivery contains no bundled active dataset or trained run artifacts.

A live browser/server test was not run in this build environment because Flask is not installed in the environment and external package installation is unavailable here.


The delivered Flask prototype completed all 20 configurations on the supplied 50,000-log synthetic dataset. The values below are measured results from the Quick profile. They are not copied from the notebook screenshots.

## Scope and settings

- This delivery starts with no bundled active dataset or trained runs. Upload a CSV and train models to generate project-specific verification artifacts.
- Supplied partitions: 35,000 training logs, 7,500 validation logs, and 7,500 held-out test logs.
- Category 1: 10 log-entry configurations; each test contains 5,625 normal and 1,875 abnormal logs.
- Category 2: 10 session configurations; each test contains 1,246 sessions: 936 normal and 310 abnormal.
- Quick profile: seed 42; 120 Random Forest trees; 5 Word2Vec epochs; up to 5 LogBERT epochs with early stopping; 100-dimensional Word2Vec; 64-dimensional LogBERT; TF-IDF caps of 20,000 features for logs and 25,000 for sessions; batch size 256.
- These Quick settings differ from the original experiment notebook. This report validates the prototype run and does not claim exact reproduction of every original notebook result.
- Thresholds and fusion weights are selected using validation data. Metrics in this report are then measured on the held-out test partition. Compare models within the same category because logs and sessions are different evaluation units.

## Checks completed

| Check | Actual result |
| --- | --- |
| Full-dataset training | All 10 log-entry and all 10 session configurations completed through the Flask API. |
| Per-record reconciliation | Every stored held-out prediction was counted by TP, FP, FN and TN for every configuration. Counts matched saved metrics and confusion-matrix totals; accuracy was recalculated from predictions. |
| Outcome filters | TP, FP and FN API filters returned totals matching the measured prediction records for all 20 configurations. |
| Download reconciliation | All 20 downloaded prediction CSVs were parsed: 87,460 configuration-specific prediction rows in total. Their outcome counts and accuracy matched the saved results. Both metrics CSVs were parsed and all 20 metric rows matched those counts. |
| Held-out example inference | Real held-out examples returned predictions from all 10 retained configurations in each category. |
| Data preparation | All 10 data unit tests passed, covering notebook cleaning, original row IDs, exact duplicates, file formats, strict labels, missing text, session eligibility, deterministic grouped splitting, transitive links and invalid splits. |
| Full cleaning compatibility | Applying the supplied cleaning rules to all 50,000 raw messages reproduced all 50,000 supplied clean_text values exactly. |
| Independent bounded engine check | All 20 configurations were exercised on a separate bounded whole-session subset. Held-out scores and decisions matched reusable inference; TP/FP/FN/TN totals reconciled. All configurations remained available for prediction after saving and reloading the model bundles. |
| Explanation methods | The bounded engine check exercised exact TF-IDF Logistic Regression contributions and token-removal explanations for another representation. |
| Unlabelled imports | JSON, TXT and CSV messages imported successfully, appeared in search, and supported actual single and batch prediction. Actual labels and TP/FP/FN/TN outcomes stayed absent when ground truth was unavailable. |
| Display limits | All / Top 100 / Top 1,000 changed the available display set while preserving the complete 50,000-row dataset. Pagination was checked. |
| Complete-dataset batch inference | The retained validation-selected log model processed all 50,000 logs. |
| Portable delivery tests | All 18 tests passed: 10 data checks and 8 saved-run checks. They verify all 87,460 saved predictions, all five requested metrics, test-only membership, reloaded inference on six examples per group, validation-based recommendations, unlabelled inputs and required session context. |

The live interface also passed isolated Microsoft Edge browser checks: the **Get started** panel, real dataset display controls, record inspection, ten predictions from a saved run, evaluation outcome filters, and comparison charts. All seven pages fit a 390-pixel viewport without page overflow. No JavaScript runtime errors or failed API responses occurred during these checks. Desktop screenshots were visually inspected.

The Windows launcher passed both paths: reusing the running application and starting a new hidden local server after the previous process was stopped. Its health endpoint confirmed successful startup at `http://127.0.0.1:5055`. Dependency installation on a second computer was not tested.

## Full-dataset test measurements

LR means Logistic Regression; RF means Random Forest. Scores below are proportions rounded to six decimal places. `test_outcomes.csv` contains the complete numeric values, thresholds, validation F1, fusion weights where applicable, timings, run identifiers and confusion counts. TP = correctly detected abnormal; FP = normal flagged abnormal; FN = missed abnormal; TN = correctly identified normal.

### Category 1: TF-IDF and Word2Vec (7,500 test logs)

Saved run: `30bfea6a3500487a`.

| Experiment | Representation | Classifier | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TF-IDF | LR | 0.997600 | 1.000000 | 0.990400 | 0.995177 | 0.000000 | 1.000000 | 1.000000 |
| 2 | TF-IDF | RF | 0.968000 | 1.000000 | 0.872000 | 0.931624 | 0.000000 | 0.999999 | 0.999997 |
| 3 | Word2Vec | LR | 0.997467 | 0.997320 | 0.992533 | 0.994921 | 0.000889 | 0.999967 | 0.999903 |
| 4 | Word2Vec | RF | 0.980933 | 1.000000 | 0.923733 | 0.960355 | 0.000000 | 0.999441 | 0.998474 |
| 5 | TF-IDF + Word2Vec concatenation | LR | 0.997467 | 0.997320 | 0.992533 | 0.994921 | 0.000889 | 0.999973 | 0.999920 |
| 6 | TF-IDF + Word2Vec concatenation | RF | 0.965600 | 1.000000 | 0.862400 | 0.926117 | 0.000000 | 0.999986 | 0.999946 |
| 7 | TF-IDF-weighted Word2Vec | LR | 0.996533 | 1.000000 | 0.986133 | 0.993018 | 0.000000 | 1.000000 | 1.000000 |
| 8 | TF-IDF-weighted Word2Vec | RF | 0.975600 | 1.000000 | 0.902400 | 0.948696 | 0.000000 | 0.999854 | 0.999531 |
| 9 | Decision fusion: TF-IDF + Word2Vec | LR | 0.997467 | 0.997320 | 0.992533 | 0.994921 | 0.000889 | 0.999967 | 0.999903 |
| 10 | Decision fusion: TF-IDF + Word2Vec | RF | 0.980933 | 1.000000 | 0.923733 | 0.960355 | 0.000000 | 0.999441 | 0.998474 |

| Experiment | Representation | Classifier | Threshold | TP | FP | FN | TN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TF-IDF | LR | 0.883613 | 1857 | 0 | 18 | 5625 |
| 2 | TF-IDF | RF | 0.741667 | 1635 | 0 | 240 | 5625 |
| 3 | Word2Vec | LR | 0.963856 | 1861 | 5 | 14 | 5620 |
| 4 | Word2Vec | RF | 0.808333 | 1732 | 0 | 143 | 5625 |
| 5 | TF-IDF + Word2Vec concatenation | LR | 0.965446 | 1861 | 5 | 14 | 5620 |
| 6 | TF-IDF + Word2Vec concatenation | RF | 0.841667 | 1617 | 0 | 258 | 5625 |
| 7 | TF-IDF-weighted Word2Vec | LR | 0.987841 | 1849 | 0 | 26 | 5625 |
| 8 | TF-IDF-weighted Word2Vec | RF | 0.883333 | 1692 | 0 | 183 | 5625 |
| 9 | Decision fusion: TF-IDF + Word2Vec | LR | 0.963856 | 1861 | 5 | 14 | 5620 |
| 10 | Decision fusion: TF-IDF + Word2Vec | RF | 0.808333 | 1732 | 0 | 143 | 5625 |

### Category 2: TF-IDF and LogBERT (1,246 test sessions)

Saved run: `d66e5734b73c48c3`.

| Experiment | Representation | Classifier | Accuracy | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TF-IDF | LR | 0.997592 | 1.000000 | 0.990323 | 0.995138 | 0.000000 | 1.000000 | 1.000000 |
| 2 | TF-IDF | RF | 0.999197 | 1.000000 | 0.996774 | 0.998384 | 0.000000 | 1.000000 | 1.000000 |
| 3 | LogBERT | LR | 0.765650 | 0.515901 | 0.941935 | 0.666667 | 0.292735 | 0.922936 | 0.805758 |
| 4 | LogBERT | RF | 0.878812 | 0.695332 | 0.912903 | 0.789400 | 0.132479 | 0.939416 | 0.771319 |
| 5 | TF-IDF + LogBERT concatenation | LR | 0.998395 | 1.000000 | 0.993548 | 0.996764 | 0.000000 | 0.999983 | 0.999949 |
| 6 | TF-IDF + LogBERT concatenation | RF | 0.997592 | 1.000000 | 0.990323 | 0.995138 | 0.000000 | 1.000000 | 1.000000 |
| 7 | TF-IDF-weighted LogBERT | LR | 0.598716 | 0.380353 | 0.974194 | 0.547101 | 0.525641 | 0.881159 | 0.673340 |
| 8 | TF-IDF-weighted LogBERT | RF | 0.884430 | 0.684444 | 0.993548 | 0.810526 | 0.151709 | 0.980997 | 0.918879 |
| 9 | Decision fusion: TF-IDF + LogBERT | LR | 0.979133 | 0.925150 | 0.996774 | 0.959627 | 0.026709 | 0.999821 | 0.999531 |
| 10 | Decision fusion: TF-IDF + LogBERT | RF | 0.976726 | 0.937695 | 0.970968 | 0.954041 | 0.021368 | 0.998208 | 0.994588 |

| Experiment | Representation | Classifier | Threshold | TP | FP | FN | TN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TF-IDF | LR | 0.935772 | 307 | 0 | 3 | 936 |
| 2 | TF-IDF | RF | 0.816667 | 309 | 0 | 1 | 936 |
| 3 | LogBERT | LR | 0.675172 | 292 | 274 | 18 | 662 |
| 4 | LogBERT | RF | 0.333333 | 283 | 124 | 27 | 812 |
| 5 | TF-IDF + LogBERT concatenation | LR | 0.780006 | 308 | 0 | 2 | 936 |
| 6 | TF-IDF + LogBERT concatenation | RF | 0.850000 | 307 | 0 | 3 | 936 |
| 7 | TF-IDF-weighted LogBERT | LR | 0.572187 | 302 | 492 | 8 | 444 |
| 8 | TF-IDF-weighted LogBERT | RF | 0.450000 | 308 | 142 | 2 | 794 |
| 9 | Decision fusion: TF-IDF + LogBERT | LR | 0.507155 | 309 | 25 | 1 | 911 |
| 10 | Decision fusion: TF-IDF + LogBERT | RF | 0.536667 | 301 | 20 | 9 | 916 |

## Complete-dataset batch result

Model `logs__tfidf__lr` from run `30bfea6a3500487a` produced 12,482 anomaly predictions and 37,518 normal predictions across all 50,000 logs. These are predicted counts, not new ground-truth class counts. This batch includes training and validation records as well as test records; it is a functional batch-processing check, not another held-out accuracy measurement.

## Data context and interpretation

The supplied data is synthetic. The audit found no exact duplicate original rows, missing cells, invalid timestamps, conflicting session labels, or scenario/session leakage across the supplied partitions. It contains 8,326 sessions, 288 template IDs and 1,999 distinct cleaned messages. There are 48,001 repeated cleaned-message rows; 959 cleaned texts and 216 templates occur in both training and test data. These repetitions are retained because they are distinct original records and are relevant when interpreting the very high TF-IDF scores.

Custom splitting was also checked on the complete dataset. It produced 35,000 / 7,500 / 7,500 records while keeping connected scenario and session groups together and both classes in every partition.

The Quick LogBERT configurations have materially different error rates from the TF-IDF baselines in this run. All measured outcomes, including false positives and missed anomalies, remain in the tables and exports.
