import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset ,random_split # Added Subset

from dataset_builder import load_raw_logs, parse_openstack_logs, generate_windows
from openstack_dataset import OpenStackDataset
from anomaly_detector import MultiSourceAnomalyDetector
from tokenizer import LogTokenizer

def train_real_data():
    print("1. Loading raw data...")
    raw_logs = load_raw_logs("./loghub/OpenStack")
    df = parse_openstack_logs(raw_logs)
    windows = generate_windows(df)
    
    with open("./loghub/OpenStack/anomaly_labels.txt", 'r') as f:
        # Strip the 'req-' prefix so the labels match our Regex parsed session_ids
        anomalous_sessions = {line.strip().replace("req-", "") for line in f.readlines() if line.strip()}
        
    print(f"Loaded {len(anomalous_sessions)} known anomalous sessions from ground truth.")
        
    tokenizer = LogTokenizer(seq_len=10)
    tokenizer.fit(df)
    
    dataset = OpenStackDataset(windows, tokenizer, anomalous_sessions)


    
   # --- REAL-WORLD CHRONOLOGICAL SPLIT ---
    # Train strictly on the past, test on the future. No peeking.
    split_idx = int(len(dataset) * 0.8)
    train_dataset = Subset(dataset, range(split_idx))
    
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    model = MultiSourceAnomalyDetector()
    embedding_layer = nn.Embedding(num_embeddings=tokenizer.idx, embedding_dim=128)
    
    # (Note: Using class weights IS standard in production for imbalanced data, 
    # but we can omit it here for the purest implementation)
    criterion = nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(embedding_layer.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    
    print(f"\nStarting training on {len(train_dataset)} windows (80% split)...")
    model.train()
    
    epochs = 3
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (traces, logs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            logs_embedded = embedding_layer(logs)
            outputs = model(traces, logs_embedded)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Window [{i+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {running_loss / len(dataloader):.4f} ---")

    # --- SAVE CHECKPOINT ---
    torch.save({
        'model_state': model.state_dict(),
        'embed_state': embedding_layer.state_dict()
    }, "openstack_checkpoint.pth")
    print("Model and embedding weights saved to openstack_checkpoint.pth")

if __name__ == "__main__":
    train_real_data()