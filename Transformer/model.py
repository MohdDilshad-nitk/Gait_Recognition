
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from typing import Dict, Tuple
# import wandb  # for logging
import random
import os
from datetime import datetime

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        """
        Rotary Positional Encoding (RoPE)
        
        Args:
            dim (int): Dimension of the embedding
        """
        super().__init__()
        self.dim = dim
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the last dimension of the input tensor
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Rotated tensor
        """
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding to the input tensor
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, dim)
        
        Returns:
            torch.Tensor: Positionally encoded tensor
        """
        seq_len = x.size(1)
        
        # Generate frequencies
        inv_freq = 1. / (10000 ** (torch.arange(0., self.dim, 2., device=x.device) / self.dim))
        
        # Create position embeddings
        position = torch.arange(seq_len, device=x.device).unsqueeze(1).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        emb = emb.to(dtype=x.dtype, device=x.device)
        
        # Reshape embeddings to match input tensor
        rotary_dim = emb.shape[-1]
        emb = emb[..., :rotary_dim]
        
        # Apply rotation to the input tensor
        rot_x = x * emb[..., :self.dim].cos() + self._rotate_half(x) * emb[..., :self.dim].sin()
        
        return rot_x



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
        # x shape: [batch_size, seq_len, num_joints, 3]
        batch_size, seq_len, num_joints, coords = x.shape

        # Combine all joints first
        x = x.reshape(batch_size, seq_len, num_joints * coords)  # [batch_size, seq_len, 60]

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
        rope: bool = False
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = SkeletonEmbedding(d_model)
        
        # self.pos_encoder = PositionalEncoding(d_model)
        self.pos_encoder = RotaryPositionalEmbedding(d_model) if rope else PositionalEncoding(d_model)

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
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_repr = self.encode(x, attention_mask)
        # projection = self.projection(sequence_repr)  # ???

        if self.classifier is not None:
            logits = self.classifier(sequence_repr)
        else:
            logits = None

        return logits
