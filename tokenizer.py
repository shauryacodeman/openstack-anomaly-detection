import pandas as pd
import torch

class LogTokenizer:
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.template_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.idx = 2
        
    def fit(self, df):
        print("Building vocabulary from unique log messages...")
        # We use the combination of component and message to define a unique log "template"
        unique_logs = (df['component'] + " " + df['message']).unique()
        
        for log in unique_logs:
            self.template_to_id[log] = self.idx
            self.idx += 1
        print(f"Vocabulary built with {self.idx} unique templates.")
            
    def transform_window(self, window_df):
        # Extract the chronological sequence of logs in this window
        sequence = []
        for _, row in window_df.head(self.seq_len).iterrows():
            log_str = row['component'] + " " + row['message']
            sequence.append(self.template_to_id.get(log_str, 1)) # 1 is <UNK>
            
        # Pad with 0s if the window has fewer logs than seq_len
        if len(sequence) < self.seq_len:
            sequence += [0] * (self.seq_len - len(sequence))
            
        return torch.tensor(sequence, dtype=torch.long)