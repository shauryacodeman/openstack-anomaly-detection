import torch
import torch.nn as nn

MODEL_NAMES = {
    "temporal": "Event order only",
    "spatial": "Service transitions only",
    "both": "Both together",
}

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # A simple 1x1 convolution to generate attention weights over the KxK matrix
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn_weights = self.sigmoid(self.conv(x))
        return x * attn_weights

class ATT_CNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=16):
        super().__init__()
        self.attention = SpatialAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU()
        # Adaptive pool ensures the output is always 2x2 regardless of the input KxK size
        self.pool = nn.AdaptiveMaxPool2d((2, 2)) 
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.attention(x)
        x = self.pool(self.relu(self.conv(x)))
        x = self.dropout(x)
        # Flatten for the fully connected layer (16 channels * 2 * 2 = 64 features)
        return torch.flatten(x, 1)

class MultiSourceAnomalyDetector(nn.Module):
    def __init__(self, seq_input_dim=128, lstm_hidden=64, cnn_out_dim=64, *, mode="both",
                 vocab_size=None, trace_channels=1):
        super().__init__()
        if mode not in MODEL_NAMES:
            raise ValueError(f"Unknown model mode: {mode}")
        self.att_cnn = None
        self.bi_lstm = None
        self.embedding = None
        features = 0
        if mode in ("spatial", "both"):
            self.att_cnn = ATT_CNN(in_channels=trace_channels)
            features += cnn_out_dim
        if mode in ("temporal", "both"):
            self.bi_lstm = nn.LSTM(seq_input_dim, lstm_hidden, bidirectional=True, batch_first=True)
            if vocab_size is not None:
                self.embedding = nn.Embedding(vocab_size, seq_input_dim, padding_idx=0)
            features += lstm_hidden * 2
        self.fc = nn.Linear(features, 2)

    def forward(self, trace_matrix, log_sequence):
        features = []
        if self.att_cnn is not None:
            features.append(self.att_cnn(trace_matrix))
        if self.bi_lstm is not None:
            valid_events = None
            if self.embedding is not None:
                valid_events = log_sequence.ne(0)
                log_sequence = self.embedding(log_sequence)
            event_features, _ = self.bi_lstm(log_sequence)
            if valid_events is not None:
                event_features = event_features.masked_fill(~valid_events.unsqueeze(-1), float("-inf"))
            pooled = event_features.amax(dim=1)
            if valid_events is not None:
                pooled = pooled.masked_fill(~valid_events.any(dim=1, keepdim=True), 0)
            features.append(pooled)
        return self.fc(torch.cat(features, dim=1))

if __name__ == "__main__":
    # --- Simulate a Forward Pass ---
    batch_size = 8
    K = 4 # 4 nodes in our distributed system
    seq_len = 10 # 10 logs per time window
    
    # Dummy trace matrices (Batch Size, Channels, Height, Width)
    mock_traces = torch.rand(batch_size, 1, K, K)
    
    # Dummy log template vectors from the previous script (Batch Size, Sequence Length, Embed Dim)
    mock_logs = torch.rand(batch_size, seq_len, 128)
    
    model = MultiSourceAnomalyDetector()
    
    # Pass both data sources through the network
    predictions = model(mock_traces, mock_logs)
    
    print(f"Input Trace Shape: {mock_traces.shape}")
    print(f"Input Log Shape: {mock_logs.shape}")
    print(f"Output Predictions Shape: {predictions.shape}")
    print(f"Raw Output Logits (Batch x 2 Classes):\n{predictions.detach().numpy()}")
