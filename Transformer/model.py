
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
import numpy as np
from typing import Tuple
from datetime import datetime


import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len: int = 5000):
        super(RotaryPositionalEmbedding, self).__init__()

        # Create a rotation matrix using tensor operations
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(1)
        j = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        self.rotation_matrix = torch.cos(i * j * 0.01)

        # Create a positional embedding matrix using tensor operations
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.arange(0, d_model, dtype=torch.float32)
        self.positional_embedding = torch.cos(position * div_term * 0.01)

        # Move to cuda if available
        self.rotation_matrix = self.rotation_matrix.cuda()
        self.positional_embedding = self.positional_embedding.cuda()

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, seq_len, d_model).

        Returns:
            A tensor of shape (batch_size, seq_len, d_model).
        """
        # Add the positional embedding to the input tensor
        seq_len = x.size(1)
        x += self.positional_embedding[:seq_len]

        # Apply the rotation matrix to the input tensor
        x = torch.matmul(x, self.rotation_matrix)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]



class SkeletonEmbedding(nn.Module):
    """New approach: combine joints first, then embed"""
    def __init__(self, d_model: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) == 4: #dealing with  gait cycles and  coordinates
          # x shape: [batch_size, seq_len, num_joints, 3]
          # Combine all joints
          batch_size, seq_len, num_joints, coords = x.shape
          x = x.reshape(batch_size, seq_len, num_joints * coords)  # [batch_size, seq_len, 60]

        else: #dealing with gait features
          # x shape: [batch_size, seq_len, gait_features(56)]
          batch_size, seq_len, feats = x.shape
          x = x.reshape(batch_size, seq_len, feats)

        # Project to d_model
        # x = self.joint_embedding(x)  # [batch_size, seq_len, d_model]  changing d_model to 60, no need to for a nn
        return x


class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        num_joints: int,
        d_model: int = 60,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = None,
        max_len: int = 5000,
        rope: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = SkeletonEmbedding(d_model)
        
        # self.pos_encoder = PositionalEncoding(d_model)
        self.pos_encoder = RotaryPositionalEmbedding(d_model = d_model,max_seq_len=max_len) if rope else PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )


        # Classification head
        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            self.classifier = None

    def encode(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x shape: [batch_size, seq_len, num_joints, 3]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        # print("embeddings shape: ", x.shape)
        x = self.pos_encoder(x)

        if attention_mask is not None:
            # Convert boolean mask to float attention mask
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(
                attention_mask == 0,
                float('-inf')
            )

        encoded = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        # Use [CLS] token (first token) as sequence representation
        sequence_repr = encoded[:, 0]

        return sequence_repr

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_repr = self.encode(x, attention_mask)
        # projection = self.projection(sequence_repr)  # ???

        if self.classifier is not None:
            logits = self.classifier(sequence_repr)
        else:
            logits = None

        if return_features:
            return logits, sequence_repr
        return logits
