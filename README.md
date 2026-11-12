# OpenStack Log Anomaly Detection 

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Machine Learning](https://img.shields.io/badge/ML-Anomaly%20Detection-brightgreen.svg)

A production-ready PyTorch implementation of a hybrid **Attention-based Convolutional Neural Network (ATT-CNN)** and **Bidirectional Long Short-Term Memory (Bi-LSTM)** architecture for detecting system anomalies in raw cloud logs. 

This pipeline processes the raw Loghub OpenStack dataset, parsing unstructured text into temporal sequences and spatial trace matrices to identify service outages and component failures.

## 🚀 Architecture Overview
System logs contain two critical dimensions of information: *what* happened (temporal sequence of messages) and *who* called whom (spatial interactions between microservices). 
* **Bi-LSTM:** Captures the temporal dependencies and sequential flow of log messages over time.
* **ATT-CNN:** Extracts the spatial relationships and structural dependencies between different system components (e.g., Nova, Neutron, Cinder) using a $K \times K$ trace matrix.
* **Dynamic Sliding Windows:** Segments logs into optimized 10-minute intervals to handle overlapping sessions without data leakage.

## 📂 Repository Structure
```text
├── dataset_builder.py     # Regex log parsing and sliding window generation
├── tokenizer.py           # Vocabulary building and log template tokenization
├── openstack_dataset.py   # PyTorch Dataset for generating trace matrices and labels
├── anomaly_detector.py    # Core ATT-CNN and Bi-LSTM neural network architecture
├── train.py               # Training loop with chronological historical split
├── evaluate.py            # Holdout evaluation script for Precision, Recall, and F1
└── requirements.txt       # Python dependencies
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shauryacodeman/openstack-anomaly-detection.git](https://github.com/shauryacodeman/openstack-anomaly-detection.git)
   cd openstack-anomaly-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   * Download the OpenStack log dataset from [Loghub (Logpai)](https://github.com/logpai/loghub).
   * Create a directory named `loghub/OpenStack/` in the root of this repository.
   * Place `openstack_normal1.log`, `openstack_normal2.log`, and `openstack_abnormal.log` into this folder.

## 🧠 Usage

**1. Train the Model**
Run the training script to ingest the logs, parse them via Regex, train the hybrid neural network on the first 80% of the chronological timeline, and save the weights.
```bash
python train.py
```
*Outputs:* `openstack_checkpoint.pth`

**2. Evaluate Performance**
Run the evaluation script to test the model against the full timeline and generate the final confusion matrix, Precision, Recall, and F1-score metrics.
```bash
python evaluate.py
```

## 👨‍💻 Author
**Shaurya Pradhan**  
Information Technology B.Tech  
[GitHub Profile](https://github.com/shauryacodeman)