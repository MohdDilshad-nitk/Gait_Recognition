import pandas as pd
import numpy as np
from pathlib import Path
import random
import re

class SkeletonDataAugmenter:
    def __init__(self, processed_data_dir, output_dir):
        """
        Initialize the skeleton data augmenter

        Args:
            processed_data_dir (str): Directory containing the processed CSV files
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir)
        self.metadata = pd.read_csv(self.processed_data_dir / 'metadata.csv')

    def normalize_to_range(self, sequence):
        """
        Normalize sequence to [0,1] range
        """
        min_vals = sequence.min()
        max_vals = sequence.max()
        normalized = (sequence - min_vals) / (max_vals - min_vals + 1e-7)
        return normalized

    def random_crop_sequence(self, sequence, min_frames=30):
        """Random continuous crop of sequence"""
        num_frames = len(sequence)
        if num_frames <= min_frames:
            return sequence

        crop_length = random.randint(min_frames, num_frames)
        start_idx = random.randint(0, num_frames - crop_length)
        return sequence.iloc[start_idx:start_idx + crop_length]

    def random_drop_frames(self, sequence, drop_ratio_range=(0.1, 0.3)):
        """Randomly drop and interpolate frames"""
        num_frames = len(sequence)
        drop_ratio = random.uniform(*drop_ratio_range)
        num_drops = int(num_frames * drop_ratio)

        if num_drops == 0:
            return sequence

        drop_indices = sorted(random.sample(range(num_frames), num_drops))
        sequence_dropped = sequence.drop(sequence.index[drop_indices])
        return sequence_dropped.interpolate(method='linear')

    def add_gaussian_noise(self, sequence, noise_level_range=(0.01, 0.03)):
        """Add random Gaussian noise to joint positions and normalize"""
        noise_level = random.uniform(*noise_level_range)
        noise = np.random.normal(0, noise_level, sequence.shape)
        noisy_sequence = pd.DataFrame(
            sequence.values + noise,
            columns=sequence.columns
        )
        return self.normalize_to_range(noisy_sequence)

    def scale_sequence(self, sequence, scale_range=(0.8, 1.2)):
        """Scale the joint positions"""
        scale = random.uniform(*scale_range)
        scaled = sequence * scale
        return self.normalize_to_range(scaled)

    def time_scale(self, sequence, scale_range=(0.8, 1.2)):
        """Scale the temporal dimension of the sequence"""
        scale = random.uniform(*scale_range)
        num_frames = len(sequence)
        new_num_frames = int(num_frames * scale)

        if new_num_frames < 3:  # Minimum frames for interpolation
            return sequence

        # Create new time points
        old_times = np.arange(num_frames)
        new_times = np.linspace(0, num_frames-1, new_num_frames)

        # Interpolate for each column
        scaled_data = np.zeros((new_num_frames, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            scaled_data[:, i] = np.interp(new_times, old_times, sequence.iloc[:, i])

        return pd.DataFrame(scaled_data, columns=sequence.columns)

    def jitter_sequence(self, sequence, jitter_range=(0.98, 1.02)):
        """Add random jitter to each joint independently"""
        jitter_factors = np.random.uniform(
            jitter_range[0],
            jitter_range[1],
            size=sequence.shape
        )
        jittered = sequence * jitter_factors
        return self.normalize_to_range(jittered)

    def get_next_sequence_number(self, person_id):
        """Get the next available sequence number for a person"""
        pattern = f"{person_id}_(\\d+)\\.csv"
        existing_files = list(self.processed_data_dir.glob(f"{person_id}_*.csv"))

        max_num = 0
        for file in existing_files:
            match = re.match(pattern, file.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

        return max_num + 1

    def validate_sequence(self, sequence):
        """
        Validate that sequence contains no negative values
        """
        return not (sequence.values < 0).any()

    def generate_augmented_sequences(self, num_augmentations=10):
        """Generate augmented sequences with various strategies"""
        augmented_metadata = []

        # Define augmentation strategies
        strategies = [
            ('crop', lambda seq: self.random_crop_sequence(seq)),
            ('drop', lambda seq: self.random_drop_frames(seq)),
            ('noise', lambda seq: self.add_gaussian_noise(seq)),
            ('scale', lambda seq: self.scale_sequence(seq)),
            ('time_scale', lambda seq: self.time_scale(seq)),
            ('jitter', lambda seq: self.jitter_sequence(seq)),
            ('crop_noise', lambda seq: self.add_gaussian_noise(self.random_crop_sequence(seq))),
            ('scale_jitter', lambda seq: self.jitter_sequence(self.scale_sequence(seq))),
            ('time_noise', lambda seq: self.add_gaussian_noise(self.time_scale(seq))),
            ('drop_scale', lambda seq: self.scale_sequence(self.random_drop_frames(seq)))
        ]

        for _, row in self.metadata.iterrows():
            # Read original sequence
            original_file = self.processed_data_dir / row['file_name']
            original_sequence = pd.read_csv(original_file)
            person_id = row['person_id']

            aug_count = 0
            max_attempts = num_augmentations * 2  # Allow some failed attempts
            attempts = 0

            #save the original file as well in output folder
            original_file_name = f"{person_id}_{row['sequence_id']}.csv"
            original_file_path = self.output_dir / original_file_name
            original_sequence.to_csv(original_file_path, index=False)

            while aug_count < num_augmentations and attempts < max_attempts:
                attempts += 1

                # Choose random augmentation strategy
                strategy_name, strategy_func = random.choice(strategies)

                # Apply augmentation
                augmented = strategy_func(original_sequence)

                # Validate sequence
                if not self.validate_sequence(augmented):
                    continue

                # Get next sequence number
                next_seq_num = self.get_next_sequence_number(person_id)

                # Create filename with incremental number
                aug_filename = f"{person_id}_{next_seq_num:02d}.csv"
                # aug_filepath = self.processed_data_dir / aug_filename
                aug_filepath = self.output_dir / aug_filename


                # Save augmented sequence
                augmented.to_csv(aug_filepath, index=False)

                # Add to metadata
                augmented_metadata.append({
                    'person_id': person_id,
                    'sequence_id': f"{next_seq_num:02d}",
                    'file_name': aug_filename,
                    'num_frames': len(augmented),
                    'augmentation_type': strategy_name,
                    'original_sequence': row['file_name']
                })

                aug_count += 1

        # Update metadata file
        augmented_metadata_df = pd.DataFrame(augmented_metadata)
        original_metadata = pd.read_csv(self.processed_data_dir / 'metadata.csv')
        updated_metadata = pd.concat([original_metadata, augmented_metadata_df], ignore_index=True)
        # updated_metadata.to_csv(self.processed_data_dir / 'metadata.csv', index=False)
        updated_metadata.to_csv(self.output_dir / 'metadata.csv', index=False)

def augment_skeleton_data(processed_data_dir, output_dir, num_augmentations=10):
    """
    Convenience function to augment skeleton data

    Args:
        processed_data_dir (str): Directory containing processed CSV files
        num_augmentations (int): Number of augmentations per sequence
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    augmenter = SkeletonDataAugmenter(processed_data_dir, output_dir)
    augmenter.generate_augmented_sequences(num_augmentations)
    return output_dir



# # After running the original SkeletonDataProcessor
# augment_skeleton_data(
#     processed_data_dir="/content/content/processed_skeleton_data",
#     num_augmentations=10
# )