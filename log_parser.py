import torch
import torch.nn as nn

class LogTemplateEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4):
        super(LogTemplateEncoder, self).__init__()
        # Layer 1: Vectorization layer (Word Embedding)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Layer 2: Transformer layer (4 Transformer encoders)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer 3: Pooling layer (Reduces sequence to a single sentence-level vector)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        embedded = self.embedding(x)  
        
        # Pass through the 4 Transformer encoders
        transformed = self.transformer_encoder(embedded) 
        
        # Permute dimensions for 1D pooling: [batch_size, embed_dim, sequence_length]
        transformed = transformed.permute(0, 2, 1)
        
        # Pool to get the final template vector: [batch_size, embed_dim]
        pooled = self.pooling(transformed).squeeze(-1) 
        
        return pooled

if __name__ == "__main__":
    # Create mock log data (Batch of 5 logs, each with 10 tokenized keywords)
    # In a real scenario, TF-IDF preprocessing converts text to these integer tokens
    vocab_size = 1000
    mock_log_tokens = torch.randint(0, vocab_size, (5, 10))
    
    # Initialize the model with 128 hidden nodes as specified in the paper
    model = LogTemplateEncoder(vocab_size=vocab_size, embed_dim=128)
    
    # Pass the logs through the model
    template_vectors = model(mock_log_tokens)
    
    print(f"Input Log Tokens Shape (Batch, Seq_Len): {mock_log_tokens.shape}")
    print(f"Output Template Vector Shape (Batch, Embed_Dim): {template_vectors.shape}")