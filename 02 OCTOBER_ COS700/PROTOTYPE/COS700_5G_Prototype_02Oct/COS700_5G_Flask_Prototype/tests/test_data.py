"""Behaviour checks for file preparation and research-safe dataset splitting."""
import io
from pathlib import Path
import sys
import unittest

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from data import apply_split, clean_log_text, parse_file, prepare_data


class DataTests(unittest.TestCase):
    def test_notebook_cleaning_and_negation(self):
        # Check notebook cleaning and negation.
        value = "2026-08-28 12:03:44.123 AMF WARNING SUPI-123 guti-abc ses-001 gnb-2 cell-3 nf-4 pol-5 sec-6 ueg-7 sst-1-sd-ABC 10.1.2.3:8080 did-not fail 97.5%"
        expected = "amf warning supi_id guti_id session_id gnb_id cell_id nf_instance_id policy_id security_context_id ue_group_id slice_id ip_address did_not fail num_value"
        self.assertEqual(clean_log_text(value), expected)

    def test_duplicates_preserve_original_positions_and_provided_text(self):
        # Check duplicates preserve original positions and provided text.
        raw = pd.DataFrame({"clean_text": [" AMF EXACT ", " AMF EXACT ", "SMF second"], "label": ["normal", "normal", "abnormal"]})
        frame, context = prepare_data(raw)
        self.assertEqual(frame["row_id"].tolist(), [1, 3])
        self.assertEqual(frame["_text"].tolist(), [" AMF EXACT ", "SMF second"])
        self.assertEqual(context["audit"]["duplicates"], 1)
        self.assertEqual(raw.shape, (3, 2))
        self.assertFalse(context["can_sessions"])
        self.assertTrue(frame["_session"].eq("").all())

    def test_unlabelled_prediction_formats(self):
        # Check unlabelled prediction formats.
        for filename, payload in [("logs.txt", b"First message\n\nSecond message\n"),
                                  ("logs.json", b'{"logs":[{"message":"First message"},{"message":"Second message"}]}'),
                                  ("logs.json", b'["First message","Second message"]'),
                                  ("logs.csv", b'message\nFirst message\nSecond message\n')]:
            frame, context = prepare_data(parse_file(io.BytesIO(payload), filename))
            self.assertEqual(len(frame), 2)
            self.assertTrue(frame["_target"].isna().all())
            self.assertFalse(context["has_labels"])
            self.assertEqual(context["source"], "User-provided dataset")

    def test_invalid_numeric_labels_and_conflicts(self):
        # Check invalid numeric labels and conflicts.
        for targets in ([0, 1.3], [0, -1], [0, float("inf")]):
            with self.assertRaises(ValueError):
                prepare_data(pd.DataFrame({"text": ["a", "b"], "target": targets}))
        with self.assertRaisesRegex(ValueError, "disagree"):
            prepare_data(pd.DataFrame({"text": ["a", "b"], "target": [0, 1], "label": ["abnormal", "abnormal"]}))
        with self.assertRaisesRegex(ValueError, "contradictory"):
            prepare_data(pd.DataFrame({"text": ["a", "a"], "target": [0, 1]}))

    def test_blank_messages_and_bad_split_fail(self):
        # Check blank messages and bad split fail.
        for message in ("", None, "!!!"):
            with self.assertRaises(ValueError):
                prepare_data(pd.DataFrame({"text": [message]}))
        with self.assertRaisesRegex(ValueError, "Unrecognised split"):
            prepare_data(pd.DataFrame({"text": ["hi"], "split": ["trian"]}))

    def test_real_sessions_and_conflicts(self):
        # Check real sessions and conflicts.
        raw = pd.DataFrame({"text": ["normal log", "error log"], "target": [0, 1], "session_id": ["A", "B"], "template_id": ["T1", "T2"]})
        self.assertTrue(prepare_data(raw)[1]["can_sessions"])
        raw.loc[1, "session_id"] = "A"
        self.assertFalse(prepare_data(raw)[1]["can_sessions"])
        raw.loc[1, "session_id"] = "B"
        raw.loc[1, "template_id"] = ""
        self.assertFalse(prepare_data(raw)[1]["can_sessions"])

    def test_custom_split_classes_groups_reproducibility(self):
        # Check custom split classes groups reproducibility.
        rows = []
        for group in range(30):
            for event in range(3):
                rows.append({"text": f"event {group} {event}", "target": group % 2,
                             "scenario_id": f"scenario-{group}", "session_id": f"session-{group}-{event // 2}", "template_id": f"T{event}"})
        frame, _ = prepare_data(pd.DataFrame(rows))
        result = apply_split(frame, "custom")
        self.assertTrue(result.equals(apply_split(frame, "custom")))
        for field in ("scenario_id", "session_id"):
            self.assertTrue(result.groupby(field)["_split"].nunique().eq(1).all())
        for name in ("train", "validation", "test"):
            self.assertEqual(set(result.loc[result["_split"].eq(name), "_target"]), {0, 1})
        self.assertTrue(frame["_split"].eq("").all())

    def test_transitive_scenario_session_links(self):
        # Check transitive scenario session links.
        rows = []
        for group in range(12):
            rows.extend([{"text": f"first {group}", "target": group % 2, "scenario_id": f"A{group}", "session_id": f"X{group}"},
                         {"text": f"second {group}", "target": group % 2, "scenario_id": f"A{group}", "session_id": f"Y{group}"},
                         {"text": f"third {group}", "target": group % 2, "scenario_id": f"B{group}", "session_id": f"Y{group}"}])
        result = apply_split(prepare_data(pd.DataFrame(rows))[0], "custom")
        for offset in range(0, len(result), 3):
            self.assertEqual(result.iloc[offset:offset + 3]["_split"].nunique(), 1)

    def test_mixed_groups_and_impossible_groups(self):
        # Check mixed groups and impossible groups.
        raw = pd.DataFrame({"text": [f"event {i}" for i in range(6)], "target": [0, 1] * 3,
                            "scenario_id": ["A", "A", "B", "B", "C", "C"]})
        result = apply_split(prepare_data(raw)[0], "custom")
        self.assertEqual(set(result["_split"]), {"train", "validation", "test"})
        raw["scenario_id"] = "ONE"
        with self.assertRaisesRegex(ValueError, "too few independent"):
            apply_split(prepare_data(raw)[0], "custom")

    def test_provided_split_leakage_and_class_requirements(self):
        # Check provided split leakage and class requirements.
        raw = pd.DataFrame({"text": [f"event {i}" for i in range(6)], "target": [0, 1] * 3,
                            "session_id": ["a", "b", "c", "d", "e", "f"], "split": ["train"] * 2 + ["validation"] * 2 + ["test"] * 2})
        apply_split(prepare_data(raw)[0])
        raw.loc[2, "session_id"] = "a"
        with self.assertRaisesRegex(ValueError, "shares scenario or session"):
            apply_split(prepare_data(raw)[0])
        raw.loc[2, "session_id"] = "c"
        raw.loc[3, "target"] = 0
        with self.assertRaisesRegex(ValueError, "validation partition"):
            apply_split(prepare_data(raw)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

