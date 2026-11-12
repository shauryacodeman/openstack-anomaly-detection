import torch
from torch.utils.data import Dataset
import pandas as pd
from trace_parser import generate_trace_matrix

class OpenStackDataset(Dataset):
    def __init__(self, windows_list, tokenizer, anomalous_sessions):
        self.windows = windows_list
        self.tokenizer = tokenizer
        self.anomalous_sessions = anomalous_sessions

    def __len__(self):
        return len(self.windows)

    def _extract_trace_edges(self, window_df):
        edges = []
        for session, group in window_df.groupby('session_id'):
            if pd.isna(session) or session in ['-', '']: continue
            components = group['component'].tolist()
            for i in range(len(components) - 1):
                edges.append({'source': components[i], 'destination': components[i+1]})
        
        if not edges:
            unique_comps = window_df['component'].unique()
            edges = [{'source': c, 'destination': c} for c in unique_comps]
            
        return pd.DataFrame(edges)

    def __getitem__(self, idx):
        window_df = self.windows[idx]
        
        # 1. Real Spatial Trace Matrix
        edges_df = self._extract_trace_edges(window_df)
        trace_tensor, _, _ = generate_trace_matrix(edges_df)
        
        # 2. Real Temporal Log Sequence
        log_sequence = self.tokenizer.transform_window(window_df)
        
        # 3. Real Label (Check if any session in this window is in the anomaly list)
        window_sessions = set(window_df['session_id'].dropna().unique())
        is_anomalous = bool(window_sessions.intersection(self.anomalous_sessions))
        label = torch.tensor(1 if is_anomalous else 0, dtype=torch.long)
        if is_anomalous:
            print(f"  -> Window {idx} flagged as ANOMALOUS (Label 1)")
        return trace_tensor, log_sequence, label