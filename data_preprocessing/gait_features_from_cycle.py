import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import gc

metadata_list = []
person_seq = {}
all_features = {}
gait_cycle_data = None

# Joint pairs and triplets
JDF_joint_pairs = [(4, 11), (5, 11), (8, 11), (9, 11), (12, 13), (8, 9), (8, 14), (9, 15), (13, 14), (13, 16), (12, 15), (12, 17), (15, 8), (15, 14), (15, 16), (17, 14), (17, 16), (9, 14)]
angle_triplets = [(2, 1, 10), (3, 1, 10), (5, 3, 1), (4, 2, 1), (3, 5, 7), (2, 4, 6), (5, 7, 9), (4, 6, 8), (13, 11, 12), (15, 13, 11), (14, 12, 11), (17, 15, 13), (16, 14, 12), (19, 17, 15), (18, 16, 14)]
BpLF_joint_pairs = [(0, 1), (1, 10), (10, 11), (12, 14), (14, 16), (13, 15), (15, 17)]
excluded_joints = {0, 1, 10, 11}  # Set for faster lookup


def save_features_chunk(output_dir, chunk_index):
    global all_features
    """Saves the current batch of processed features and clears memory."""
    chunk_file = os.path.join(output_dir, f"data_{chunk_index}.pkl")
    with open(chunk_file, 'wb') as f:
        pickle.dump(all_features, f)

# def extract_gait_features_optimized(df):
#     """Efficiently extracts gait features using vectorized operations."""
#     distances = {}
#     data_np = df.to_numpy().reshape(df.shape[0], -1, 3)  # Shape: (num_frames, num_joints, 3)
    
#     # Compute distances using vectorized operations
#     for joint1, joint2 in BpLF_joint_pairs + JDF_joint_pairs:
#         distances[f'Distance_{joint1}_{joint2}'] = np.linalg.norm(data_np[:, joint1] - data_np[:, joint2], axis=1)
    
#     # Compute angles using vectorized operations
#     for j1, j2, j3 in angle_triplets:
#         u = data_np[:, j1] - data_np[:, j2]
#         v = data_np[:, j3] - data_np[:, j2]
        
#         dot_product = np.einsum('ij,ij->i', u, v)
#         mag_u = np.linalg.norm(u, axis=1)
#         mag_v = np.linalg.norm(v, axis=1)
        
#         with np.errstate(invalid='ignore', divide='ignore'):
#             cosine_angles = np.clip(dot_product / (mag_u * mag_v), -1.0, 1.0)
#             angles = np.arccos(cosine_angles)
        
        
#         distances[f'Angle_{j1}_{j2}_{j3}'] = np.degrees(angles) / 360
    
#     # Compute inter-frame distances
#     joints = [i for i in range(20) if i not in excluded_joints]
#     inter_distances = np.linalg.norm(np.diff(data_np[:, joints], axis=0), axis=2)
#     inter_distances = np.vstack([inter_distances, inter_distances[-1]])  # Repeat last row to match frame count
    
#     for idx, joint in enumerate(joints):
#         distances[f'InterFrameDistance_Joint_{joint}'] = inter_distances[:, idx]
    
#     return pd.DataFrame(distances)


def extract_gait_features_optimized(df):
    """Efficiently extracts gait features using vectorized operations."""
    distances = {}
    data_np = df.to_numpy().reshape(df.shape[0], -1, 3)  # Shape: (num_frames, num_joints, 3)

    # ✅ Compute distances using vectorized operations
    for joint1, joint2 in BpLF_joint_pairs + JDF_joint_pairs:
        distances[f'Distance_{joint1}_{joint2}'] = np.linalg.norm(data_np[:, joint1] - data_np[:, joint2], axis=1)

    # ✅ Compute angles using vectorized operations
    epsilon = 1e-6
    for j1, j2, j3 in angle_triplets:
        u = data_np[:, j1] - data_np[:, j2]
        v = data_np[:, j3] - data_np[:, j2]

        # Dot product and magnitudes
        # dot_product = np.sum(u * v, axis=1)
        dot_product = np.einsum('ij,ij->i', u, v)
        mag_u = np.sqrt(np.sum(u**2, axis=1))
        mag_v = np.sqrt(np.sum(v**2, axis=1))

        # Valid mask to avoid zero division
        valid = (mag_u > epsilon) & (mag_v > epsilon)
        cosine_angles = np.zeros_like(dot_product)
        angles = np.zeros_like(dot_product)

        # Compute angles where valid
        cosine_angles[valid] = dot_product[valid] / (mag_u[valid] * mag_v[valid])
        cosine_angles = np.clip(cosine_angles, -1.0, 1.0)
        angles[valid] = np.arccos(cosine_angles[valid])

        distances[f'Angle_{j1}_{j2}_{j3}'] = np.degrees(angles) / 360

    # ✅ Compute inter-frame distances
    joints = [i for i in range(20) if i not in excluded_joints]
    inter_distances = np.linalg.norm(np.diff(data_np[:, joints], axis=0), axis=2)

    # Pad last row to match frame count
    last_row = inter_distances[-1] if len(inter_distances) > 0 else np.zeros(len(joints))
    inter_distances = np.vstack([inter_distances, last_row])

    for idx, joint in enumerate(joints):
        distances[f'InterFrameDistance_Joint_{joint}'] = inter_distances[:, idx]

    # if any(np.any(np.isnan(val)) for val in distances.values()):
    #     print("found nan")

    # ✅ FINAL NaN handling — replace all NaNs at once (fast and memory-efficient)
    distances_df = pd.DataFrame(distances).fillna(0)

    return distances_df

def extract_gait_features_from_cycles(input_dir, output_dir):
    """Processes all gait cycle data and saves optimized feature extractions."""
    print("\nStarting batch-optimized gait feature extraction...")
    print("input directory: ", input_dir, ", output directory: ", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    global all_features, gait_cycle_data

    input_file_path = os.path.join(input_dir, 'data.pkl')
    

    # Load chunk data
    with open(input_file_path, 'rb') as f:
        gait_cycle_data = pickle.load(f)

   
    for filename, df in tqdm(gait_cycle_data.items()):
        if not filename.endswith('.csv') or filename == 'metadata.csv':
            continue

        input_file = filename[:-4]
        features_df = extract_gait_features_optimized(df)

        # output_file = os.path.join(output_dir, f"{input_file}.csv")
        all_features[f"{input_file}.csv"] = features_df
    

        # Metadata handling
        person_id = input_file[:9]
        person_seq[person_id] = person_seq.get(person_id, 0) + 1

        metadata_list.append({
            'person_id': person_id,
            'sequence_id': person_seq[person_id],
            'file_name': f"{input_file}.csv",
            'num_frames': len(features_df),
            
        })

    
    # Save metadata
    pd.DataFrame(metadata_list).to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    os.remove(input_file_path)

    with open(os.path.join(output_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(all_features, f)
    
    print("Batch-optimized gait feature extraction completed.\n")
    return output_dir
