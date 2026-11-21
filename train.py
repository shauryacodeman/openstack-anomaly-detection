import argparse

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from anomaly_detector import MODEL_NAMES, MultiSourceAnomalyDetector
from dataset_builder import load_experiment_split
from openstack_dataset import WindowDataset, openstack_examples
from rcaeval_dataset import load_rcaeval_split
from tokenizer import LogTokenizer

SEED = 42
EPOCHS = 8
BATCH_SIZE = 8
VALIDATION_FALSE_ALARM_LIMIT = 0.30
CHECKPOINT = "openstack_experiment_v2.pth"
CHECKPOINTS = {"openstack": CHECKPOINT, "rcaeval": "rcaeval_re3ob_neural.pth"}


def dataset_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=CHECKPOINTS, default="openstack")
    return parser.parse_args().dataset


def load_windows(name):
    if name == "rcaeval":
        return load_rcaeval_split()
    train, test = load_experiment_split()
    return openstack_examples(train), openstack_examples(test)


def fit_inputs(train_windows, dataset_name):
    train_events = pd.concat([window.events for window in train_windows], ignore_index=True)
    tokenizer = LogTokenizer(seq_len=512)
    tokenizer.fit(train_events)
    if dataset_name == "rcaeval":
        nodes = set()
        for window in train_windows:
            nodes.update(window.edges["source"].unique())
            nodes.update(window.edges["destination"].unique())
    else:
        nodes = set(train_events["component"])
    if not nodes:
        raise ValueError("Training needs at least one service in the call graph.")
    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}
    return tokenizer, node_to_index


def fit_model(mode, dataset, vocab_size, device, *, verbose=True):
    counts = dataset.labels.bincount(minlength=2)
    if (counts == 0).any():
        raise ValueError("Training needs both normal and anomalous windows.")

    weights = len(dataset) / (2 * counts.float())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), reduction="none")
    torch.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = MultiSourceAnomalyDetector(
        mode=mode, vocab_size=vocab_size, trace_channels=dataset.traces[0].shape[0]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for traces, logs, labels in loader:
            traces, logs, labels = traces.to(device), logs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(traces, logs), labels).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
        if verbose:
            print(f"  Epoch {epoch}/{EPOCHS} | training loss {total_loss / len(dataset):.4f}")
    return model


@torch.inference_mode()
def anomaly_scores(model, dataset, device):
    model.eval()
    return torch.cat([
        model(traces.to(device), logs.to(device)).softmax(dim=1)[:, 1].cpu()
        for traces, logs, _ in DataLoader(dataset, batch_size=BATCH_SIZE)
    ])


def alarm_threshold(scores, labels):
    thresholds = torch.cat((scores, scores.new_tensor([0.5]), (scores.max() + 1e-6).view(1))).unique()
    predicted = scores >= thresholds[:, None]
    positive = labels.bool()
    true_positives = (predicted & positive).sum(dim=1)
    false_positives = (predicted & ~positive).sum(dim=1)
    allowed = false_positives <= VALIDATION_FALSE_ALARM_LIMIT * (~positive).sum()
    best_recall = true_positives[allowed].max()
    # Higher cutoffs break recall ties in favor of fewer false alarms.
    return thresholds[allowed & (true_positives == best_recall)].max().item()


def calibrate_rcaeval(train_windows, device):
    scores = {mode: [] for mode in MODEL_NAMES}
    labels = []
    for repetition in (1, 2):
        validation = [window for window in train_windows if window.run_id.endswith(f"_{repetition}")]
        fitting = [window for window in train_windows if not window.run_id.endswith(f"_{repetition}")]
        print(f"Calibrating on repetition {repetition} using the other training repetition...")
        tokenizer, nodes = fit_inputs(fitting, "rcaeval")
        training = WindowDataset(fitting, tokenizer, nodes)
        held_out = WindowDataset(validation, tokenizer, nodes)
        reference = training.healthy_trace_reference()
        training.apply_trace_reference(reference)
        held_out.apply_trace_reference(reference)
        labels.append(held_out.labels)
        for mode in MODEL_NAMES:
            model = fit_model(mode, training, tokenizer.idx, device, verbose=False)
            scores[mode].append(anomaly_scores(model, held_out, device))
    return {mode: alarm_threshold(torch.cat(values), torch.cat(labels))
            for mode, values in scores.items()}


def train_real_data(dataset_name="openstack"):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")

    checkpoint_path = CHECKPOINTS[dataset_name]
    train_windows, _ = load_windows(dataset_name)
    thresholds = calibrate_rcaeval(train_windows, device) if dataset_name == "rcaeval" else {}
    tokenizer, node_to_index = fit_inputs(train_windows, dataset_name)
    dataset = WindowDataset(train_windows, tokenizer, node_to_index)
    reference = None
    if dataset_name == "rcaeval":
        reference = dataset.healthy_trace_reference()
        dataset.apply_trace_reference(reference)
    normal, anomalous = dataset.labels.bincount(minlength=2).tolist()
    print(f"Training set: {normal} normal and {anomalous} anomalous windows.")

    model_states = {}
    for mode, name in MODEL_NAMES.items():
        print(f"\nTraining: {name}")
        model = fit_model(mode, dataset, tokenizer.idx, device)
        model_states[mode] = {name: value.cpu() for name, value in model.state_dict().items()}

    torch.save({
        "models": model_states,
        "templates": tokenizer.template_to_id,
        "sequence_length": tokenizer.seq_len,
        "nodes": node_to_index,
        "trace_channels": dataset.traces[0].shape[0],
        "thresholds": thresholds,
        "trace_reference": reference,
    }, checkpoint_path)
    print(f"\nSaved all three models to {checkpoint_path}.")


if __name__ == "__main__":
    train_real_data(dataset_argument())
