import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
import os

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

        # self.min_vals, self.max_vals = self.overall_min_max()
        self.all_sequences = {}

    def overall_min_max(self):
        """Same as your original implementation"""
        min_vals = np.inf * np.ones((self.num_joints, 3))
        max_vals = -np.inf * np.ones((self.num_joints, 3))

        for person_dir in self.person_dirs:
            skeleton_files = sorted(person_dir.glob('*.txt'))

            for file_path in skeleton_files:
                sequence = self.read_skeleton_file(file_path)
                min_vals = np.minimum(min_vals, sequence.min(axis=(0, 1)))
                max_vals = np.maximum(max_vals, sequence.max(axis=(0, 1)))

        return min_vals, max_vals

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

        # min_vals = self.min_vals
        # max_vals = self.max_vals
        normalized_sequence = (normalized_sequence - min_vals) / (max_vals - min_vals + 1e-7)

        return normalized_sequence

    def save_processed_data(self, output_path):
        """
        Process all skeleton data and save to CSV files

        Args:
            output_dir (str): Directory to save processed data
        """
    
        os.makedirs(output_path, exist_ok=True)

        # Create metadata list
        metadata_list = []

        data_file_paths = []
        for person_dir in self.person_dirs:
            person_id = person_dir.name
            skeleton_files = sorted(person_dir.glob('*.txt'))

            for file_path in skeleton_files:
                data_file_paths.append([person_id, file_path])


        for [person_id, file_path] in tqdm(data_file_paths):
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
            output_file = f'{person_id}_{sequence_id}.csv'
            
            # sequence_df.to_csv(output_file, index=False)

            self.all_sequences[output_file] = sequence_df

            # Add to metadata
            metadata_list.append({
                'person_id': person_id,
                'sequence_id': sequence_id,
                'file_name': f'{person_id}_{sequence_id}.csv',
                'num_frames': len(sequence)
            })

        # Save metadata
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.to_csv(output_path + '/metadata.csv', index=False)

        with open(output_path + '/data.pkl', 'wb') as f:
          pickle.dump(self.all_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
          
        del self.all_sequences

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
    processed_data_dir = processor.save_processed_data(output_path=output_dir)
    print("Transformation completed...\n\n")
    return processed_data_dir



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