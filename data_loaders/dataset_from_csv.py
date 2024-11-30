import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

class SkeletonDatasetFromCSV(Dataset):
    def __init__(self, data_dir, split_metadata_file):
        """
        Create a PyTorch dataset from processed CSV files for a specific split

        Args:
            data_dir (str): Directory containing processed CSV files and metadata
            split_metadata_file (str): Name of the split metadata file (e.g., 'train_metadata.csv')
        """
        self.data_dir = Path(data_dir)

        # Read split-specific metadata
        self.metadata = pd.read_csv(self.data_dir / split_metadata_file)

        # Create person ID mapping
        self.person_ids = sorted(self.metadata['person_id'].unique())
        self.person_to_idx = {pid: i for i, pid in enumerate(self.person_ids)}

        # Set max sequence length
        self.max_len = self.metadata['num_frames'].max()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get metadata for this sequence
        row = self.metadata.iloc[idx]

        # Read sequence from CSV
        sequence_df = pd.read_csv(self.data_dir / row['file_name'])
        sequence = sequence_df.values

        # Reshape back to (frames, joints, 3)
        sequence = sequence.reshape(sequence.shape[0], -1, 3)

        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence)

        # Pad sequence if necessary
        if sequence_tensor.size(0) < self.max_len:
            padding = torch.zeros(self.max_len - sequence_tensor.size(0),
                                sequence_tensor.size(1),
                                sequence_tensor.size(2))
            sequence_tensor = torch.cat([sequence_tensor, padding], dim=0)

        # Create attention mask
        attention_mask = torch.ones(self.max_len)
        attention_mask[sequence.shape[0]:] = 0

        return {
            'person_id': self.person_to_idx[row['person_id']],
            'sequence_id': row['sequence_id'],
            'sequence': sequence_tensor,
            'attention_mask': attention_mask
        }

