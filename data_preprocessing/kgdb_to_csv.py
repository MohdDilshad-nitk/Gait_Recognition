import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class SkeletonDataProcessor:
    def __init__(self, data_dir):
        """
        Initialize the skeleton data processor

        Args:
            data_dir (str): Path to the root directory containing person folders
        """
        self.data_dir = Path(data_dir)
        self.person_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.num_joints = 20

    def read_skeleton_file(self, file_path):
        """Same as your original implementation"""
        with open(file_path, 'r') as file:
            lines = file.read().strip().split('\n')

        num_frames = len(lines) // self.num_joints
        sequence = np.zeros((num_frames, self.num_joints, 3))

        for i, line in enumerate(lines):
            parts = line.split(';')
            frame_idx = i // self.num_joints
            joint_idx = i % self.num_joints
            sequence[frame_idx, joint_idx] = [float(parts[1]), float(parts[2]), float(parts[3])]

        return sequence

    def normalize_sequence(self, sequence):
        """Same as your original implementation"""
        hip_center_idx = 11
        normalized_sequence = sequence.copy()

        for frame_idx in range(sequence.shape[0]):
            hip_center = sequence[frame_idx, hip_center_idx]
            normalized_sequence[frame_idx] -= hip_center

        min_vals = normalized_sequence.min(axis=(0, 1), keepdims=True)
        max_vals = normalized_sequence.max(axis=(0, 1), keepdims=True)
        normalized_sequence = (normalized_sequence - min_vals) / (max_vals - min_vals + 1e-7)

        return normalized_sequence

    def save_processed_data(self, output_dir):
        """
        Process all skeleton data and save to CSV files

        Args:
            output_dir (str): Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create metadata list
        metadata_list = []

        for person_dir in tqdm(self.person_dirs):
            person_id = person_dir.name
            skeleton_files = sorted(person_dir.glob('*.txt'))

            for file_path in skeleton_files:
                sequence_id = file_path.stem

                # Read and normalize sequence
                sequence = self.read_skeleton_file(file_path)
                normalized_sequence = self.normalize_sequence(sequence)

                # Reshape sequence to 2D for saving to CSV
                # Shape: (frames, joints*3)
                flattened_sequence = normalized_sequence.reshape(normalized_sequence.shape[0], -1)

                # Create column names
                columns = [f'joint_{j}_{coord}' for j in range(self.num_joints)
                          for coord in ['x', 'y', 'z']]

                # Save sequence to CSV
                sequence_df = pd.DataFrame(flattened_sequence, columns=columns)
                output_file = output_path / f'{person_id}_{sequence_id}.csv'
                sequence_df.to_csv(output_file, index=False)

                # Add to metadata
                metadata_list.append({
                    'person_id': person_id,
                    'sequence_id': sequence_id,
                    'file_name': f'{person_id}_{sequence_id}.csv',
                    'num_frames': len(sequence)
                })

        # Save metadata
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.to_csv(output_path / 'metadata.csv', index=False)

        return output_path


def process_skeleton_data(data_dir, output_dir):
    """
    Process and save skeleton data

    Args:
        data_dir (str): Directory containing person folders
        output_dir (str): Directory to save processed data
    """

    print("\n\nStarting transformation...")
    print("input directory: ", data_dir, ", output directory: ", output_dir)
    processor = SkeletonDataProcessor(data_dir=data_dir)
    processed_data_dir = processor.save_processed_data(output_dir=output_dir)
    print("Transformation completed...\n\n")
    return processed_data_dir

# # Step 1: Process and save the data
# processor = SkeletonDataProcessor(data_dir="/content/Data")
# processed_data_dir = processor.save_processed_data(output_dir="processed_skeleton_data")


'''
Head 0
Shoulder-Center 1
Shoulder-Right 2
Shoulder-Left 3
Elbow-Right 4
Elbow-Left 5
Wrist-Right 6
Wrist-Left 7
Hand-Right 8
Hand-Left 9
Spine 10
Hip-centro 11
Hip-Right 12
Hip-Left 13
Knee-Right 14
Knee-Left 15
Ankle-Right 16
Ankle-Left 17
Foot-Right 18
Foot-Left 19

'''