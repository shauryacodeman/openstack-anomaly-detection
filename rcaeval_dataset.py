from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from openstack_dataset import TelemetryWindow

DATA_DIR = Path(__file__).with_name("data") / "rcaeval"
WINDOW_SECONDS = 300
GUARD_SECONDS = 60
SPAN_COLUMNS = [
    "startTimeMillis", "traceID", "spanID", "parentSpanID", "serviceName", "duration", "statusCode",
]
VARIABLE_FIELDS = re.compile(
    r"(?P<field>\bproduct_ids=)\[[^\]\r\n]+\]"
    r"|(?P<currency>\b[A-Z]{3})(?=\d)\d+(?:\.\d+)?"
    r"|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"
    r"|@[0-9a-fA-F]{6,16}\b"
    r"|\b(?=[A-Z0-9]{10}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{10}\b"
)


def normalize_messages(messages):
    def replace(match):
        if match["field"]:
            return match["field"] + "[<id>]"
        if match["currency"]:
            return match["currency"] + "<id>"
        return "<id>"

    return messages.str.replace(VARIABLE_FIELDS, replace, regex=True)


def split_cases(cases):
    by_fault = defaultdict(dict)
    for case in cases:
        group, repeat = case.name.rsplit("_", 1)
        by_fault[group][repeat] = case

    train, test = [], []
    for group, repeats in sorted(by_fault.items()):
        if repeats.keys() != {"1", "2", "3"}:
            raise ValueError(f"{group} needs all three recorded repetitions.")
        train.extend((repeats["1"], repeats["2"]))
        test.append(repeats["3"])
    return train, test


def span_edges(spans):
    spans = spans.dropna(subset=["traceID", "spanID", "serviceName"])
    spans = spans.drop_duplicates(["traceID", "spanID"])
    parents = spans[["traceID", "spanID", "serviceName"]].rename(
        columns={"spanID": "parentSpanID", "serviceName": "source"}
    )
    calls = spans.merge(parents, on=["traceID", "parentSpanID"], validate="many_to_one")
    calls = calls.loc[calls["source"] != calls["serviceName"]].rename(columns={"serviceName": "destination"})
    return calls.assign(
        latency_ms=calls["duration"].clip(lower=0) / 1000,
        error=calls["statusCode"].ne(0).astype("Float32"),
    )[["source", "destination", "latency_ms", "error"]]


def case_windows(case):
    injected_at = int((case / "inject_time.txt").read_text(encoding="utf-8").strip())
    logs = pd.read_parquet(case / "logs.parquet", columns=["timestamp", "container_name", "message"])
    spans = pd.read_parquet(case / "traces.parquet", columns=SPAN_COLUMNS)
    logs = logs.rename(columns={"container_name": "component"}).dropna(subset=["component", "message"])
    logs = logs.sort_values("timestamp", kind="stable")

    intervals = (
        (injected_at - GUARD_SECONDS - WINDOW_SECONDS, injected_at - GUARD_SECONDS),
        (injected_at + GUARD_SECONDS, injected_at + GUARD_SECONDS + WINDOW_SECONDS),
    )
    windows = []
    for label, (start, end) in enumerate(intervals):
        events = logs.loc[logs["timestamp"].between(start, end, inclusive="left")]
        edges = span_edges(spans.loc[spans["startTimeMillis"].between(start * 1000, end * 1000, inclusive="left")])
        if events.empty or edges.empty:
            raise ValueError(f"{case.name} has incomplete telemetry for label {label}.")
        events = events.assign(message=normalize_messages(events["message"]))
        windows.append(TelemetryWindow(events, edges, label, case.name))
    return windows


def load_rcaeval_split(data_dir=DATA_DIR):
    cases = sorted(case for case in Path(data_dir).glob("re3ob_*") if case.is_dir())
    if len(cases) != 30:
        raise ValueError(f"Expected 30 RCAEval RE3-OB runs in {data_dir}. See the README for download commands.")

    train, test = split_cases(cases)
    print(f"RCAEval RE3-OB: reading {len(train)} training runs and {len(test)} held-out runs...")
    return (
        [window for case in train for window in case_windows(case)],
        [window for case in test for window in case_windows(case)],
    )
