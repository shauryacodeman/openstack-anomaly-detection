from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset

from trace_parser import generate_trace_matrix


@dataclass
class TelemetryWindow:
    events: pd.DataFrame
    edges: pd.DataFrame
    label: int
    run_id: str | None = None


class WindowDataset(Dataset):
    def __init__(self, windows, tokenizer, node_to_index):
        self.labels = torch.tensor([window.label for window in windows], dtype=torch.long)
        self.traces = [generate_trace_matrix(window.edges, node_to_index)[0] for window in windows]
        self.logs = [tokenizer.transform_window(window.events) for window in windows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.traces[idx], self.logs[idx], self.labels[idx]

    def healthy_trace_reference(self):
        normal = torch.stack([trace for trace, label in zip(self.traces, self.labels) if label == 0])
        return normal.mean(dim=0), normal.std(dim=0, unbiased=False).clamp_min(0.01)

    def apply_trace_reference(self, reference):
        mean, scale = reference
        self.traces = [((trace - mean) / scale).clamp(-5, 5) for trace in self.traces]


def openstack_examples(windows):
    examples = []
    for window in windows:
        destination = window.groupby("session_id")["component"].shift(-1)
        edges = pd.DataFrame({"source": window["component"], "destination": destination}).dropna()
        label = int(window["fault_vm"].notna().any())
        examples.append(TelemetryWindow(window, edges[edges["source"] != edges["destination"]], label))
    return examples


class OpenStackDataset(WindowDataset):
    def __init__(self, windows, tokenizer, node_to_index):
        super().__init__(openstack_examples(windows), tokenizer, node_to_index)
