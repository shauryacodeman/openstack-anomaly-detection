import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from anomaly_detector import MultiSourceAnomalyDetector
from dataset_builder import LOG_FILES, generate_windows, load_experiment_split
from evaluate import score_predictions
from openstack_dataset import OpenStackDataset, WindowDataset
from rcaeval_dataset import case_windows, normalize_messages, split_cases
from tokenizer import LogTokenizer
from train import alarm_threshold
from trace_parser import generate_trace_matrix


class ExperimentTest(unittest.TestCase):
    def test_rcaeval_templates_and_alarm_cutoff(self):
        messages = pd.Series([
            "needed product_ids=['66VCHSJNUP', '0PUK6V6EV0'] for nick@example.org",
            "needed product_ids=['LS4PSXUNUM'] for casey@example.net",
            "Exception at Handler@783565e1",
            "Exception at Handler@71e28fb4",
        ])
        self.assertEqual(normalize_messages(messages).tolist(), [
            "needed product_ids=[<id>] for <id>",
            "needed product_ids=[<id>] for <id>",
            "Exception at Handler<id>",
            "Exception at Handler<id>",
        ])
        scores = torch.tensor([0.8, 0.7, 0.6, 0.5])
        labels = torch.tensor([1, 0, 0, 1])
        self.assertAlmostEqual(alarm_threshold(scores, labels), 0.8)
        self.assertGreater(alarm_threshold(torch.full((4,), 0.5), labels), 0.5)

    def test_rcaeval_split_and_recorded_calls(self):
        cases = [Path(f"re3ob_mail_f1_{i}") for i in (3, 1, 2)]
        self.assertEqual(split_cases(cases), ([cases[1], cases[2]], [cases[0]]))

        logs = pd.DataFrame({
            "timestamp": [900, 1000, 1100],
            "container_name": ["mail"] * 3,
            "message": ["before fault", "during injection", "after fault"],
        })
        spans = pd.DataFrame({
            "startTimeMillis": [900000, 900001, 1100000, 1100001, 1100001],
            "traceID": ["a", "a", "b", "b", "other-trace"],
            "spanID": ["parent", "child", "parent", "child", "different-child"],
            "parentSpanID": [None, "parent", None, "parent", "parent"],
            "serviceName": ["front", "mail", "front", "mail", "wrong"],
            "duration": [0, 1000, 0, 7000, 1000],
            "statusCode": [0, 0, 0, 12, 0],
        })
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "inject_time.txt").write_text("1000")
            with patch("rcaeval_dataset.pd.read_parquet", side_effect=[logs, spans]):
                normal, anomalous = case_windows(case)

        self.assertEqual([normal.label, anomalous.label], [0, 1])
        self.assertEqual(normal.events["message"].tolist(), ["before fault"])
        self.assertEqual(anomalous.events["message"].tolist(), ["after fault"])
        self.assertEqual(anomalous.edges[["source", "destination"]].values.tolist(), [["front", "mail"]])
        matrix, *_ = generate_trace_matrix(anomalous.edges, {"front": 0, "mail": 1})
        self.assertEqual(matrix.shape, (3, 2, 2))
        self.assertEqual(matrix[2, 0, 1].item(), 1)
        dataset = WindowDataset([normal, anomalous], LogTokenizer(), {"front": 0, "mail": 1})
        dataset.apply_trace_reference(dataset.healthy_trace_reference())
        self.assertEqual(torch.count_nonzero(dataset[0][0]).item(), 0)
        self.assertGreater(dataset[1][0][2, 0, 1].item(), 0)

    def test_window_extension_keeps_nearby_events_only(self):
        events = pd.DataFrame({
            "timestamp": pd.to_datetime(["2017-05-14 21:00", "2017-05-14 21:11", "2017-05-14 21:13"]),
            "session_id": ["same-request"] * 3,
        })
        self.assertEqual(generate_windows(events)[0].index.tolist(), [0, 1])

    def test_metrics_count_hits_misses_and_false_alarms(self):
        labels = torch.tensor([0, 0, 0, 1, 1])
        predictions = torch.tensor([0, 1, 1, 0, 1])
        self.assertEqual(score_predictions(labels, predictions), (1, 1, 2, 1, 1 / 3, 0.5, 0.4))
        self.assertEqual(score_predictions(labels, torch.zeros_like(labels)), (0, 2, 0, 3, 0, 0, 0))

    def test_window_labels_holdout_and_templates(self):
        train_vm = "544fd51c-4edc-4780-baae-ba1d80a0acfc"
        test_vm = "a445709b-6ad0-40ec-8860-bec60b6ca0c2"
        request = "d6143bd4-9784-4e95-a0c8-83fba00530b6"

        def event(time, component, message):
            return f"{time} 1 INFO {component} [req-{request}] {message}\n"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "anomaly_labels.txt").write_text(f"Affected VM IDs\n{train_vm}\n{test_vm}\n")
            (root / LOG_FILES[0]).write_text(event("2017-05-16 00:00:00.000", "compute", "Created VM 123"))
            (root / LOG_FILES[1]).write_text(event("2017-05-17 00:00:00.000", "compute", "Created VM 456"))
            (root / LOG_FILES[2]).write_text(
                event("2017-05-14 21:24:00.000", "compute", f"Deleted {train_vm}_del")
                + event("2017-05-14 21:25:00.000", "compute", f"Deleted {test_vm}_del")
                + event("2017-05-14 21:26:00.000", "network", "Test only message")
            )
            train, test = load_experiment_split(root)

        self.assertEqual(set().union(*(set(window["fault_vm"].dropna()) for window in train)), {train_vm})
        self.assertEqual(set().union(*(set(window["fault_vm"].dropna()) for window in test)), {test_vm})
        tokenizer = LogTokenizer(seq_len=4)
        tokenizer.fit(train[0])
        nodes = {"compute": 0, "network": 1}
        self.assertEqual(OpenStackDataset(train, tokenizer, nodes).labels.tolist(), [0, 1])
        dataset = OpenStackDataset(test, tokenizer, nodes)
        self.assertEqual(dataset.labels.tolist(), [0, 1])
        self.assertEqual(dataset[0][1].tolist(), [2, 0, 0, 0])
        self.assertEqual(dataset[1][1].tolist(), [1, 1, 0, 0])
        self.assertEqual(dataset[1][0][0, 0, 1].item(), 1)

        traces, logs, _ = dataset[1]
        temporal = MultiSourceAnomalyDetector(mode="temporal", vocab_size=tokenizer.idx)
        spatial = MultiSourceAnomalyDetector(mode="spatial", vocab_size=tokenizer.idx)
        self.assertEqual(temporal(None, logs.unsqueeze(0)).shape, (1, 2))
        self.assertTrue(torch.isfinite(temporal(None, torch.zeros(1, 4, dtype=torch.long))).all())
        self.assertEqual(spatial(traces.unsqueeze(0), None).shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
