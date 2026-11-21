import pandas as pd
import numpy as np
import torch

def generate_trace_matrix(trace_df, node_to_index=None):
    if node_to_index is None:
        nodes = pd.concat([trace_df["source"], trace_df["destination"]]).unique()
        node_to_index = {node: i for i, node in enumerate(nodes)}
    K = len(node_to_index)

    if "latency_ms" in trace_df:
        calls = trace_df.groupby(["source", "destination"]).agg(
            count=("latency_ms", "size"), latency=("latency_ms", "mean"), error=("error", "mean")
        ).fillna(0)
        # Use a fixed log scale for counts and latency so magnitude is comparable across windows.
        calls[["count", "latency"]] = np.log1p(calls[["count", "latency"]]) / 10
        features = np.zeros((3, K, K), dtype=np.float32)
        for (source, destination), values in calls.iterrows():
            if source in node_to_index and destination in node_to_index:
                features[:, node_to_index[source], node_to_index[destination]] = values.to_numpy(dtype=np.float32)
        return torch.from_numpy(features), K, node_to_index
    
    # 2. Initialize an empty K x K matrix (using NumPy first for easy indexing)
    # The paper colors areas based on call rate, frequency, and delay. 
    # We will start with a single channel: call frequency.
    trace_matrix = np.zeros((K, K), dtype=np.float32)
    
    # 3. Aggregate the data (count total calls between each node pair)
    call_counts = trace_df.value_counts(["source", "destination"])
    
    # 4. Populate the matrix
    for (source, destination), count in call_counts.items():
        if source in node_to_index and destination in node_to_index:
            trace_matrix[node_to_index[source], node_to_index[destination]] = count
        
    # Normalize the matrix to act like a grayscale image (values between 0 and 1)
    max_count = trace_matrix.max(initial=0)
    if max_count > 0:
        trace_matrix /= max_count
        
    # Convert to a PyTorch tensor with shape [Channels, Height, Width]
    # Since it's a 2D matrix, we add a channel dimension of 1.
    tensor_image = torch.tensor(trace_matrix).unsqueeze(0)
    
    return tensor_image, K, node_to_index

if __name__ == "__main__":
    # Create mock OpenStack trace data to test the pipeline
    data = {
        'source': ['nova-api', 'nova-api', 'neutron-server', 'nova-compute', 'nova-api'],
        'destination': ['nova-compute', 'neutron-server', 'nova-compute', 'cinder-volume', 'nova-compute'],
        'success': [1, 1, 1, 0, 1],
        'delay_ms': [12.5, 8.2, 15.1, 110.3, 10.1]
    }
    df = pd.DataFrame(data)
    
    tensor_img, K, mapping = generate_trace_matrix(df)
    
    print(f"Total Nodes (K): {K}")
    print(f"Node Mapping: {mapping}")
    print(f"Output Tensor Shape: {tensor_img.shape}")
    print(f"Matrix Output:\n{tensor_img.squeeze().numpy()}")
