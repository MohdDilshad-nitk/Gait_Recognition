
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

# class TemporalAugmenter:
#     def __init__(
#         self,
#         crop_ratio_range=(0.8, 0.9),
#         mask_ratio_range=(0.1, 0.2),
#         min_sequence_length=16
#     ):
#         """
#         Initialize temporal augmentation parameters

#         Args:
#             crop_ratio_range (tuple): Range for random crop ratio
#             mask_ratio_range (tuple): Range for random masking ratio
#             min_sequence_length (int): Minimum sequence length after cropping
#         """
#         self.crop_ratio_range = crop_ratio_range
#         self.mask_ratio_range = mask_ratio_range
#         self.min_sequence_length = min_sequence_length

#     def random_temporal_crop(
#         self,
#         sequence: torch.Tensor,
#         attention_mask: torch.Tensor = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Apply random temporal cropping to sequence"""
#         seq_length = sequence.size(1)

#         # Determine crop size
#         min_ratio, max_ratio = self.crop_ratio_range
#         crop_ratio = random.uniform(min_ratio, max_ratio)
#         crop_size = max(int(seq_length * crop_ratio), self.min_sequence_length)

#         # Random start point
#         max_start = seq_length - crop_size
#         start_idx = random.randint(0, max_start)
#         end_idx = start_idx + crop_size

#         # Apply crop
#         cropped_sequence = sequence[:, start_idx:end_idx]

#         if attention_mask is not None:
#             cropped_mask = attention_mask[:, start_idx:end_idx]
#             return cropped_sequence, cropped_mask

#         return cropped_sequence, None

#     def random_temporal_mask(
#         self,
#         sequence: torch.Tensor,
#         attention_mask: torch.Tensor = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Apply random temporal masking to sequence"""
#         seq_length = sequence.size(1)

#         # Determine number of segments to mask
#         min_ratio, max_ratio = self.mask_ratio_range
#         mask_ratio = random.uniform(min_ratio, max_ratio)
#         num_masks = max(1, int(seq_length * mask_ratio))

#         # Create copy of sequence for masking
#         masked_sequence = sequence.clone()
#         if attention_mask is not None:
#             new_attention_mask = attention_mask.clone()

#         # Apply random masks
#         for _ in range(num_masks):
#             # Random mask length between 1 and 5 frames
#             mask_length = random.randint(1, min(5, seq_length // 10))
#             start_idx = random.randint(0, seq_length - mask_length)
#             end_idx = start_idx + mask_length

#             # Apply mask (set to zeros)
#             masked_sequence[:, start_idx:end_idx] = 0

#             if attention_mask is not None:
#                 new_attention_mask[:, start_idx:end_idx] = 0

#         if attention_mask is not None:
#             return masked_sequence, new_attention_mask

#         return masked_sequence, None



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
        num_classes: int = None
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = SkeletonEmbedding(d_model)
        # print(self.embedding.shape)
        self.pos_encoder = PositionalEncoding(d_model)

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
