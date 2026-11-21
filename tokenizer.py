import re

import torch

from dataset_builder import UUID

VARIABLE = re.compile(rf"{UUID}|\b[0-9a-f]{{32}}\b|\b\d+(?:\.\d+)*\b", re.IGNORECASE)


class LogTokenizer:
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.template_to_id = {"<PAD>": 0, "<UNK>": 1}

    @property
    def idx(self):
        return len(self.template_to_id)

    @staticmethod
    def templates(df):
        return df["component"] + " " + df["message"].str.replace(VARIABLE, "<id>", regex=True)

    def fit(self, df):
        for template in self.templates(df.drop_duplicates(["component", "message"])).unique():
            self.template_to_id.setdefault(template, self.idx)
        print(f"Training vocabulary: {self.idx - 2} log templates.")

    def transform_window(self, window_df):
        if len(window_df) > self.seq_len:
            # shortcut: cap dense sequences by sampling across the window; use full sequences for event-level recall.
            positions = [i * len(window_df) // self.seq_len for i in range(self.seq_len)]
            window_df = window_df.iloc[positions]
        sequence = [self.template_to_id.get(template, 1) for template in self.templates(window_df)]
        sequence.extend([0] * (self.seq_len - len(sequence)))
        return torch.tensor(sequence, dtype=torch.long)
