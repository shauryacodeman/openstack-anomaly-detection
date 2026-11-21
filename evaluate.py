import torch

from anomaly_detector import MODEL_NAMES, MultiSourceAnomalyDetector
from openstack_dataset import WindowDataset
from tokenizer import LogTokenizer
from train import CHECKPOINTS, anomaly_scores, dataset_argument, load_windows


def score_predictions(labels, predictions):
    # Encode (actual, predicted) as 0 for true negatives, 1 for false positives, 2 for misses and 3 for hits.
    tn, fp, fn, tp = (2 * labels + predictions).bincount(minlength=4).tolist()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return tp, fn, fp, tn, precision, recall, f1


@torch.inference_mode()
def evaluate_model(dataset_name="openstack"):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")
    checkpoint = torch.load(CHECKPOINTS[dataset_name], map_location="cpu", weights_only=True)
    tokenizer = LogTokenizer(seq_len=checkpoint["sequence_length"])
    tokenizer.template_to_id = checkpoint["templates"]

    _, test_windows = load_windows(dataset_name)
    dataset = WindowDataset(test_windows, tokenizer, checkpoint["nodes"])
    if checkpoint.get("trace_reference") is not None:
        dataset.apply_trace_reference(checkpoint["trace_reference"])
    normal, anomalous = dataset.labels.bincount(minlength=2).tolist()
    if not normal or not anomalous:
        raise ValueError("Testing needs both normal and anomalous windows.")
    print(f"Test set: {normal} normal and {anomalous} anomalous windows.")

    prediction_rows = []
    for mode, name in MODEL_NAMES.items():
        print(f"Testing: {name}")
        model = MultiSourceAnomalyDetector(
            mode=mode, vocab_size=tokenizer.idx, trace_channels=checkpoint.get("trace_channels", 1)
        ).to(device)
        model.load_state_dict(checkpoint["models"][mode])
        scores = anomaly_scores(model, dataset, device)
        threshold = checkpoint.get("thresholds", {}).get(mode)
        if threshold is None:
            predictions = scores.gt(0.5).long()
        else:
            print(f"  Validation cutoff: {threshold:.3f}")
            predictions = scores.ge(threshold).long()
        prediction_rows.append((name, predictions))

    print("\nHeld-out test results (counts are windows)")
    print(f"{'Model':<26} {'Caught':>6} {'Missed':>6} {'False alarms':>12} {'Correct normal':>14} {'Precision':>10} {'Recall':>7} {'F1':>7}")
    for name, predictions in prediction_rows:
        tp, fn, fp, tn, precision, recall, f1 = score_predictions(dataset.labels, predictions)
        print(f"{name:<26} {tp:>6} {fn:>6} {fp:>12} {tn:>14} {precision:>10.3f} {recall:>7.3f} {f1:>7.3f}")


if __name__ == "__main__":
    evaluate_model(dataset_argument())
