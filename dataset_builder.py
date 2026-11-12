import re
from datetime import datetime

import pandas as pd
import os
from datetime import timedelta

def load_raw_logs(data_dir):
    """Loads and combines the three split OpenStack log files."""
    log_files = [
        'openstack_normal1.log',
        'openstack_normal2.log',
        'openstack_abnormal.log'
    ]
    
    raw_lines = []
    for file in log_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file}...")
        with open(file_path, 'r') as f:
            raw_lines.extend(f.readlines())
            
    print(f"Total log lines loaded: {len(raw_lines)}")
    return raw_lines

def parse_openstack_logs(raw_lines):
    print("Parsing logs with Regex...")
    # Removed the '^' start anchor so it ignores the Loghub file prefixes
    log_pattern = re.compile(
        r'(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+'
        r'(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[\w\.]+)\s+'
        r'\[(?:req-)?(?P<session_id>[\w\-]+)?.*?\]\s+(?P<message>.*)'
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
                
    df = pd.DataFrame(parsed_data)
    print(f"Successfully extracted {len(df)} valid log events.")
    return df

def generate_windows(df, window_size_mins=10, step_size_mins=5, max_expansion_mins=30):
    print(f"Segmenting data into optimized {window_size_mins}-minute windows...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    if df.empty: return []
        
    start_time = df['timestamp'].iloc[0]
    end_time = df['timestamp'].iloc[-1]
    
    windows = []
    current_time = start_time
    
    while current_time < end_time:
        window_end = current_time + timedelta(minutes=window_size_mins)
        
        # 1. Base sliding window
        mask = (df['timestamp'] >= current_time) & (df['timestamp'] < window_end)
        base_window = df[mask]
        
        # 2. Optimized Window Expansion (with safety bounds)
        if not base_window.empty:
            # Filter out empty or placeholder session IDs (like '-' or None)
            valid_sessions = base_window[~base_window['session_id'].isin([None, '', '-'])]['session_id'].unique()
            
            # Cap the lookahead to prevent massive timeline jumps
            max_lookahead = window_end + timedelta(minutes=max_expansion_mins)
            
            future_mask = (
                (df['timestamp'] >= window_end) & 
                (df['timestamp'] <= max_lookahead) & 
                (df['session_id'].isin(valid_sessions))
            )
            
            expanded_window = pd.concat([base_window, df[future_mask]])
            windows.append(expanded_window)
        
        current_time += timedelta(minutes=step_size_mins)
        
    print(f"Generated {len(windows)} optimized windows.")
    return windows

if __name__ == "__main__":
    data_directory = "./loghub/OpenStack"
    
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