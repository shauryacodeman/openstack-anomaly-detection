import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset , random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from dataset_builder import load_raw_logs, parse_openstack_logs, generate_windows
from openstack_dataset import OpenStackDataset
from anomaly_detector import MultiSourceAnomalyDetector
from tokenizer import LogTokenizer

def evaluate_model():
    print("1. Rebuilding pipeline for evaluation...")
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
    
    # ... [Keep your imports and data loading] ...
    dataset = OpenStackDataset(windows, tokenizer, anomalous_sessions)
    
    # --- FULL DATASET AUDIT ---
    # Evaluate the entire timeline to view the true confusion matrix
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"2. Initializing model for evaluation on all {len(dataset)} windows...")
    model = MultiSourceAnomalyDetector()
    embedding_layer = nn.Embedding(num_embeddings=tokenizer.idx, embedding_dim=128)
    
    # Load the trained weights
    checkpoint = torch.load("openstack_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state'])
    embedding_layer.load_state_dict(checkpoint['embed_state'])
    
    model.eval()
    embedding_layer.eval()
    
    all_preds = []
    all_labels = []
    
    print("3. Running predictions...")
    with torch.no_grad():
        for traces, logs, labels in eval_dataloader: # Pass full eval_dataloader
            logs_embedded = embedding_layer(logs)
            outputs = model(traces, logs_embedded)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.append(predicted.item())
            all_labels.append(labels.item())
            
    # ... [Keep your sklearn metrics code at the bottom] ...
            
    # Compute paper metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print("\n=== FINAL EVALUATION METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    if conf_matrix.shape == (2, 2):
        print(f"True Normal: {conf_matrix[0][0]}    | False Positive: {conf_matrix[0][1]}")
        print(f"False Negative: {conf_matrix[1][0]} | True Anomalous: {conf_matrix[1][1]}")
    else:
        print(conf_matrix)

if __name__ == "__main__":
    evaluate_model()