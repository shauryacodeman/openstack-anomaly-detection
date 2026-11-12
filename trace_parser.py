import pandas as pd
import numpy as np
import torch

def generate_trace_matrix(trace_df):
    # 1. Extract unique nodes to determine K (matrix size)
    nodes = pd.concat([trace_df['source'], trace_df['destination']]).unique()
    K = len(nodes)
    
    # Create a mapping from node name to matrix index (0 to K-1)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    # 2. Initialize an empty K x K matrix (using NumPy first for easy indexing)
    # The paper colors areas based on call rate, frequency, and delay. 
    # We will start with a single channel: call frequency.
    trace_matrix = np.zeros((K, K), dtype=np.float32)
    
    # 3. Aggregate the data (count total calls between each node pair)
    call_counts = trace_df.groupby(['source', 'destination']).size().reset_index(name='count')
    
    # 4. Populate the matrix
    for _, row in call_counts.iterrows():
        src_idx = node_to_index[row['source']]
        dst_idx = node_to_index[row['destination']]
        trace_matrix[src_idx, dst_idx] = row['count']
        
    # Normalize the matrix to act like a grayscale image (values between 0 and 1)
    if trace_matrix.max() > 0:
        trace_matrix = trace_matrix / trace_matrix.max()
        
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