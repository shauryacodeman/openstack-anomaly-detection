import torch
import torch.nn as nn

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
    def __init__(self, seq_input_dim=128, lstm_hidden=64, cnn_out_dim=64):
        super().__init__()
        self.att_cnn = ATT_CNN()
        self.bi_lstm = nn.LSTM(seq_input_dim, lstm_hidden, bidirectional=True, batch_first=True)
        
        # Bi-LSTM outputs 2 * hidden_size (forward and backward directions)
        combined_dim = cnn_out_dim + (lstm_hidden * 2)
        
        # Binary classification: 0 (Normal) or 1 (Anomalous)
        self.fc = nn.Linear(combined_dim, 2) 

    def forward(self, trace_matrix, log_sequence):
        # 1. Extract spatial features from the KxK trace matrix
        cnn_features = self.att_cnn(trace_matrix)
        
        # 2. Extract temporal features from the log sequence
        lstm_out, (hn, cn) = self.bi_lstm(log_sequence)
        
        # Get the final hidden state from both the forward and backward LSTM passes
        lstm_features = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        
        # 3. Concatenate spatial and temporal features and classify
        combined_features = torch.cat((cnn_features, lstm_features), dim=1)
        logits = self.fc(combined_features)
        
        return logits

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