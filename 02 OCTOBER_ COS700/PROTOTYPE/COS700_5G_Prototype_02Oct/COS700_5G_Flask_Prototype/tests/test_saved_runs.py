"""Verify the delivered experiments without repeating full model training."""

# Resolve all artifacts relative to this portable project.
import json
import os
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from engine import predict_bundle, train_experiment

INSTANCE = Path(os.environ.get("COS700_INSTANCE", PROJECT / "instance"))


class SavedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reload the newest completed artifact for each experiment group.
        cls.runs = {}
        for path in sorted((INSTANCE / "runs").glob("*/run.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            group = metadata["config"]["group"]
            previous = cls.runs.get(group)
            if previous is None or metadata["created_at"] > previous["metadata"]["created_at"]:
                cls.runs[group] = {"metadata": metadata, "directory": path.parent}
        if set(cls.runs) != {"logs", "sessions"}:
            raise unittest.SkipTest("No bundled trained runs are included; the prototype now starts blank and runs are created after CSV upload and training.")
        for run in cls.runs.values():
            metadata, directory = run["metadata"], run["directory"]
            run["bundle"] = joblib.load(directory / "models.joblib")
            run["frame"] = pd.read_pickle(INSTANCE / "datasets" / f"{metadata['dataset_id']}.pkl")
            run["partitions"] = pd.read_csv(directory / "partitions.csv", dtype={"row_id": str})
            run["predictions"] = {
                result["model_id"]: pd.read_csv(directory / f"{result['model_id']}.csv",
                                               dtype={"row_id": str, "unit_id": str, "session_id": str},
                                               keep_default_na=False, float_precision="round_trip")
                for result in metadata["results"]
            }

    def test_all_configurations_and_encoders_are_persisted(self):
        # Require every configuration, including the reusable session encoder.
        for group, run in self.runs.items():
            with self.subTest(group=group):
                metadata, bundle = run["metadata"], run["bundle"]
                self.assertEqual(len(metadata["results"]), 10)
                self.assertEqual(set(bundle["models"]), set(run["predictions"]))
                self.assertEqual(set(bundle["predictions"]), set(run["predictions"]))
                self.assertEqual(bundle["group"], group)
                self.assertGreater(bundle["feature_info"]["vocabulary_size"], 0)
                if group == "sessions":
                    self.assertIsNotNone(bundle["features"].encoder)
                    self.assertGreater(len(bundle["features"].vocabulary), 4)
                    self.assertGreater(len(bundle["training_history"]), 0)
                else:
                    self.assertIsNotNone(bundle["features"].w2v)

    def test_all_saved_outcomes_and_metrics_reconcile(self):
        # Check every documented test record against its threshold and metrics.
        for group, run in self.runs.items():
            for result in run["metadata"]["results"]:
                with self.subTest(group=group, model=result["model_id"]):
                    rows = run["predictions"][result["model_id"]]
                    actual, predicted = rows.actual.to_numpy(), rows.predicted.to_numpy()
                    np.testing.assert_array_equal(predicted, (rows.score >= result["threshold"]).astype(int))
                    labels = np.select([(actual == 1) & (predicted == 1), (actual == 0) & (predicted == 1),
                                        (actual == 1) & (predicted == 0)], ["TP", "FP", "FN"], default="TN")
                    np.testing.assert_array_equal(rows.outcome.to_numpy(), labels)
                    counts = rows.outcome.value_counts()
                    tp, fp, fn, tn = (int(counts.get(key, 0)) for key in ("TP", "FP", "FN", "TN"))
                    self.assertEqual(result["confusion_matrix"], [[tn, fp], [fn, tp]])
                    self.assertEqual([result[key] for key in ("tp", "fp", "fn", "tn")], [tp, fp, fn, tn])
                    self.assertEqual(result["test_count"], len(rows))
                    expected = {"accuracy": (tp + tn) / len(rows), "precision": tp / (tp + fp) if tp + fp else 0,
                                "recall": tp / (tp + fn) if tp + fn else 0,
                                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                                "fpr": fp / (fp + tn) if fp + tn else 0}
                    for metric, value in expected.items():
                        self.assertAlmostEqual(result[metric], value, places=10)

    def test_evidence_contains_only_the_saved_test_partition(self):
        # Exclude training or validation records from reported test evidence.
        for group, run in self.runs.items():
            partitions, frame = run["partitions"], run["frame"]
            test_ids = set(partitions.loc[partitions._split.eq("test"), "row_id"])
            frame_test = frame.loc[frame.row_id.astype(str).isin(test_ids)]
            expected_units = set(frame_test.row_id.astype(str)) if group == "logs" else set(frame_test._session.astype(str))
            for model_id, rows in run["predictions"].items():
                with self.subTest(group=group, model=model_id):
                    self.assertEqual(set(rows.unit_id), expected_units)
                    self.assertTrue(set(rows.row_id).issubset(test_ids))
                    self.assertFalse(rows.unit_id.duplicated().any())
            related = frame[["row_id", "_session"]].copy()
            related["row_id"] = related.row_id.astype(str)
            related = related.merge(partitions, on="row_id", validate="one_to_one")
            known = related.loc[related._session.notna() & related._session.ne("")]
            self.assertTrue(known.groupby("_session")._split.nunique().eq(1).all())

    def test_reloaded_inference_matches_multiple_heldout_units(self):
        # Rescore several normal and anomalous units using reloaded models.
        for group, run in self.runs.items():
            baseline = next(iter(run["predictions"].values()))
            examples = pd.concat([baseline.loc[baseline.actual.eq(label)].head(3) for label in (0, 1)])
            frame = run["frame"]
            selected = frame.loc[frame.row_id.astype(str).isin(examples.row_id)] if group == "logs" else frame.loc[frame._session.astype(str).isin(examples.unit_id)]
            output = pd.DataFrame(predict_bundle(run["bundle"], selected, session=group == "sessions"))
            self.assertEqual(len(output), len(examples) * 10)
            for model_id, rows in run["predictions"].items():
                with self.subTest(group=group, model=model_id):
                    fresh = output.loc[output.model_id.eq(model_id)].set_index("unit_id").sort_index()
                    saved = rows.loc[rows.unit_id.isin(examples.unit_id)].set_index("unit_id").sort_index()
                    self.assertEqual(fresh.index.tolist(), saved.index.tolist())
                    np.testing.assert_allclose(fresh.score, saved.score, atol=1e-5, rtol=1e-5)
                    np.testing.assert_array_equal(fresh.predicted, saved.predicted)
                    np.testing.assert_array_equal(fresh.outcome, saved.outcome)

    def test_default_model_selection_uses_validation_f1(self):
        # Confirm the default recommendation never ranks test performance.
        for group, run in self.runs.items():
            metadata = run["metadata"]
            selected = next(result for result in metadata["results"] if result["model_id"] == metadata["recommended_model_id"])
            with self.subTest(group=group):
                self.assertEqual(selected["validation_f1"], max(result["validation_f1"] for result in metadata["results"]))
                self.assertEqual(metadata["selection_basis"], "Highest validation F1")

    def test_unlabelled_inference_has_no_invented_outcomes(self):
        # Keep confidence separate from unknown ground truth.
        run = self.runs["logs"]
        inputs = run["frame"].head(2).copy()
        inputs["_target"] = np.nan
        rows = predict_bundle(run["bundle"], inputs)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["actual"] is None and row["outcome"] is None for row in rows))
        without_labels = predict_bundle(run["bundle"], inputs.drop(columns="_target"))
        np.testing.assert_allclose([row["score"] for row in rows], [row["score"] for row in without_labels])

    def test_blank_session_ids_allow_log_training(self):
        # Exercise the ungrouped-upload regression with twelve real records.
        frame = self.runs["logs"]["frame"]
        tiny = frame.groupby(["_split", "_target"], group_keys=False).head(2).copy()
        tiny["_session"] = ""
        bundle = train_experiment(tiny, {"group": "logs", "representation": "tfidf", "classifier": "lr"})
        self.assertEqual(len(bundle["results"]), 1)
        self.assertEqual(bundle["results"][0]["test_count"], 4)

    def test_session_models_require_complete_session_identifiers(self):
        # Reject isolated or unidentified records for session inference.
        run = self.runs["sessions"]
        with self.assertRaisesRegex(ValueError, "complete session"):
            predict_bundle(run["bundle"], "amf normal log")
        inputs = run["frame"].head(2).copy()
        inputs["_session"] = ""
        with self.assertRaisesRegex(ValueError, "genuine session ID"):
            predict_bundle(run["bundle"], inputs, session=True)


# Run the same checks directly or through unittest discovery.
if __name__ == "__main__":
    unittest.main(verbosity=2)
