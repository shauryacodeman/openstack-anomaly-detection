import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).with_name("loghub") / "OpenStack"
LOG_FILES = ("openstack_normal1.log", "openstack_normal2.log", "openstack_abnormal.log")
UUID = r"(?<![0-9a-f])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?![0-9a-f])"


def load_raw_logs(data_dir, log_files=LOG_FILES):
    """Loads and combines the three split OpenStack log files."""
    raw_lines = []
    for file in log_files:
        print(f"Loading {file}...")
        with (Path(data_dir) / file).open(encoding="utf-8") as f:
            raw_lines.extend(f.readlines())
            
    print(f"Total log lines loaded: {len(raw_lines)}")
    return raw_lines

def parse_openstack_logs(raw_lines):
    print("Parsing logs with Regex...")
    # Removed the '^' start anchor so it ignores the Loghub file prefixes
    log_pattern = re.compile(
        r'(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+'
        r'(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[\w\.]+)\s+'
        r'\[(?:req-(?P<session_id>[\w][\w-]*))?.*?\]\s+(?P<message>.*)'
    )
    
    parsed_data = []
    for line in raw_lines:
        # Changed from .match() to .search() to look inside the string
        match = log_pattern.search(line.strip())
        if match:
            data = match.groupdict()
            ts_str = f"{data['date']} {data['time']}"
            try:
                data['timestamp'] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                parsed_data.append(data)
            except ValueError:
                continue
                
    df = pd.DataFrame(parsed_data, columns=[*log_pattern.groupindex, "timestamp"])
    print(f"Successfully extracted {len(df)} valid log events.")
    return df

def mark_fault_events(df, vm_ids):
    # Match whole VM IDs in messages, including IDs followed by a path suffix such as _del.
    ids = "|".join(sorted(vm_ids))
    df = df.copy()
    df["fault_vm"] = df["message"].str.extract(
        rf"(?i)(?<![0-9a-f])({ids})(?![0-9a-f])", expand=False
    ).str.lower()
    return df


def load_experiment_split(data_dir=DATA_DIR):
    labels = (Path(data_dir) / "anomaly_labels.txt").read_text(encoding="utf-8")
    vm_ids = {line.strip().lower() for line in labels.splitlines()
              if re.fullmatch(UUID, line.strip(), re.IGNORECASE)}
    if not vm_ids:
        raise ValueError("No VM UUIDs found in anomaly_labels.txt.")

    train_normal, test_normal, abnormal = [
        mark_fault_events(parse_openstack_logs(load_raw_logs(data_dir, (filename,))), vm_ids)
        for filename in LOG_FILES
    ]
    # shortcut: this cutoff targets the gap in the official Loghub run; set a new split for another capture.
    cutoff = pd.Timestamp("2017-05-14 21:25:00")
    before_cutoff = abnormal["timestamp"] < cutoff
    train_faults, test_faults = abnormal[before_cutoff], abnormal[~before_cutoff]
    train_vms = set(train_faults["fault_vm"].dropna())
    test_vms = set(test_faults["fault_vm"].dropna())
    if (
        not train_vms or not test_vms or train_vms & test_vms
        or (train_vms | test_vms) != vm_ids
        or train_normal["fault_vm"].notna().any() or test_normal["fault_vm"].notna().any()
    ):
        raise ValueError("Expected separate groups of labeled VMs in train and test, and two normal captures.")

    return (
        generate_windows(train_normal) + generate_windows(train_faults),
        generate_windows(test_normal) + generate_windows(test_faults),
    )


def generate_windows(df, window_size_mins=10, step_size_mins=5, max_expansion_mins=2):
    print(f"Segmenting data into optimized {window_size_mins}-minute windows...")
    if df.empty:
        return []

    df = df.sort_values('timestamp', kind='stable')
    timestamps = df['timestamp']
    start_time = timestamps.iloc[0]
    end_time = timestamps.iloc[-1]
    
    windows = []
    current_time = start_time
    
    while current_time <= end_time:
        window_end = current_time + timedelta(minutes=window_size_mins)
        
        # 1. Base sliding window
        start_idx = timestamps.searchsorted(current_time)
        end_idx = timestamps.searchsorted(window_end)
        base_window = df.iloc[start_idx:end_idx]
        
        # 2. Optimized Window Expansion (with safety bounds)
        if not base_window.empty:
            # Filter out empty or placeholder session IDs (like '-' or None)
            valid_sessions = base_window[~base_window['session_id'].isin([None, '', '-'])]['session_id'].unique()
            
            # Reused background request IDs can connect logs many minutes apart.
            max_lookahead = window_end + timedelta(minutes=max_expansion_mins)
            
            lookahead_idx = timestamps.searchsorted(max_lookahead, side='right')
            future = df.iloc[end_idx:lookahead_idx]
            expanded_window = pd.concat([base_window, future[future['session_id'].isin(valid_sessions)]])
            windows.append(expanded_window)
        
        current_time += timedelta(minutes=step_size_mins)
        
    print(f"Generated {len(windows)} optimized windows.")
    return windows

if __name__ == "__main__":
    data_directory = DATA_DIR
    
    # 1. Load and merge
    merged_logs = load_raw_logs(data_directory)
    
    # 2. Extract structured data
    structured_df = parse_openstack_logs(merged_logs)
    
    # 3. Generate optimized windows
    time_windows = generate_windows(structured_df)
    
    # 4. Preview the first window
    if time_windows:
        first_window = time_windows[0]
        print(f"\nFirst Window Size: {len(first_window)} events")
        print(f"First Window Time Range: {first_window['timestamp'].min()} to {first_window['timestamp'].max()}")
