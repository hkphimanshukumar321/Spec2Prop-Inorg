import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class RamanFormer1D(nn.Module):
    """
    State-of-the-art Hybrid CNN-Transformer model for Raman Spectroscopy.
    1. 1D CNN Tokenizer: Extracts local peak features and downsamples the spectrum.
    2. Transformer Encoder: Learns global long-range chemical relationships.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = None,
                 d_model: int = 128, nhead: int = 4, num_encoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.3):
        super().__init__()
        
        # 1. CNN Tokenizer (Front-end)
        # Reduces 2048 sequence length to 128 (factor of 16)
        self.tokenizer = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(64, d_model, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        
        # Positional Encoding for sequence order
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=256)
        
        # 2. Transformer Encoder (Back-end)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 3. Classification Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        ) if num_classes else None
        
    def forward_features(self, x):
        # x: (batch, 1, 2048)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # CNN Feature Extraction
        x = self.tokenizer(x) # shape: (batch, d_model, seq_len)
        
        # Prepare for Transformer (batch_first=True)
        x = x.permute(0, 2, 1) # shape: (batch, seq_len, d_model)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer Global Modeling
        x = self.transformer_encoder(x)
        
        # Global Average Pooling over the sequence
        x = x.mean(dim=1) # shape: (batch, d_model)
        
        return x

    def forward(self, x):
        emb = self.forward_features(x)
        if self.head is not None:
            return self.head(emb)
        return emb
