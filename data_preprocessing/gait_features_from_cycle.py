import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

metadata_list = []
person_seq = {}

# Note all the values are zero based indexed, but in the paper its one based indexed
JDF_joint_pairs = [
    (4,11),
    (5,11),
    (8,11),
    (9,11),
    (12,13),
    (8,9),
    (8,14),
    (9,15),
    (13,14),
    (13,16),
    (12,15),
    (12,17),
    (15,8),
    (15,14),
    (15,16),
    (17,14),
    (17,16),
    (9,14)
]

angle_triplets = [(2, 1, 10), 
                  (3, 1, 10), 
                  (5, 3, 1), 
                  (4, 2, 1), 
                  (3, 5, 7), 
                  (2, 4, 6), 
                  (5, 7, 9), 
                  (4, 6, 8), 
                  (13, 11, 12), 
                  (15, 13, 11), 
                  (14, 12, 11), 
                  (17, 15, 13), 
                  (16, 14, 12), 
                  (19, 17, 15), 
                  (18, 16, 14)]

BpLF_joint_pairs = [
        (0, 1),  # Head to Shoulder-Center
        (1, 10), # Shoulder-Center to Spine
        (10, 11), # Spine to Hip-Centro
        (12, 14), # Hip-Right to Knee-Right
        (14, 16), # Knee-Right to Ankle-Right
        (13, 15), # Hip-Left to Knee-Left
        (15, 17), # Knee-Left to Ankle-Left
    ]

def calculate_angle(j1, j2, j3, data):
    # Vectors
    u = np.array([data[f'joint_{j1}_x'] - data[f'joint_{j2}_x'],
                  data[f'joint_{j1}_y'] - data[f'joint_{j2}_y'],
                  data[f'joint_{j1}_z'] - data[f'joint_{j2}_z']])
    v = np.array([data[f'joint_{j3}_x'] - data[f'joint_{j2}_x'],
                  data[f'joint_{j3}_y'] - data[f'joint_{j2}_y'],
                  data[f'joint_{j3}_z'] - data[f'joint_{j2}_z']])
    
    # Dot product and magnitudes
    dot_product = np.einsum('ij,ij->j', u, v)
    mag_u = np.linalg.norm(u, axis=0)
    mag_v = np.linalg.norm(v, axis=0)

    # Handle zero magnitudes
    valid_indices = (mag_u > 1e-6) & (mag_v > 1e-6)  # Avoid division by zero
    angles = np.zeros_like(dot_product)  # Default to 0 degrees
    angles[valid_indices] = np.arccos(np.clip(dot_product[valid_indices] / (mag_u[valid_indices] * mag_v[valid_indices]), -1.0, 1.0))

    # Convert to degrees and normalize
    angles_normalized = np.degrees(angles) / 360

    return angles_normalized

def extract_gait_features(input_file, output_dir):
    """
    Extracts gait cycle features from a CSV file and saves them as separate CSV files.

    Parameters:
    input_file (str): Path to the input directory.
    output_dir (str): Path to the directory where the extracted gait cycle CSV files will be saved.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the input CSV file
    df = pd.read_csv(input_file)
    distances = {}

    # BpLF
    for joint1, joint2 in BpLF_joint_pairs:
        x1, y1, z1 = df[f'joint_{joint1}_x'], df[f'joint_{joint1}_y'], df[f'joint_{joint1}_z']
        x2, y2, z2 = df[f'joint_{joint2}_x'], df[f'joint_{joint2}_y'], df[f'joint_{joint2}_z']
        distances[f'Distance_{joint1}_{joint2}'] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    #JDF
    for joint1, joint2 in JDF_joint_pairs:
        x1, y1, z1 = df[f'joint_{joint1}_x'], df[f'joint_{joint1}_y'], df[f'joint_{joint1}_z']
        x2, y2, z2 = df[f'joint_{joint2}_x'], df[f'joint_{joint2}_y'], df[f'joint_{joint2}_z']
        distances[f'Distance_{joint1}_{joint2}'] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


    #JAF
    for j1, j2, j3 in angle_triplets:
      distances[f'Angle_{j1}_{j2}_{j3}'] = calculate_angle(j1, j2, j3, df)


    #INjdf
    excluded_joints = [0, 1, 10, 11]
    joints = [i for i in range(20) if i not in excluded_joints]

    for joint in joints:
      x = df[f'joint_{joint}_x'].values
      y = df[f'joint_{joint}_y'].values
      z = df[f'joint_{joint}_z'].values

      # Compute distances between consecutive frames
      inter_distances = np.sqrt((np.diff(x))**2 + (np.diff(y))**2 + (np.diff(z))**2)

      # Append the last row distance as the last value
      last_distance = inter_distances[-1]
      inter_distances = np.append(inter_distances, last_distance)

      # Store the distances in the dictionary
      distances[f'InterFrameDistance_Joint_{joint}'] = inter_distances

    distances_df = pd.DataFrame(distances)

    # Save the extracted gait features as separate CSV files
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.csv")

    # Add to metadata
    person_id = base_name[0:9]
    if person_id not in person_seq:
        person_seq[person_id] = 0

    person_seq[person_id] += 1
    metadata_list.append({
        'person_id': person_id,
        'sequence_id': person_seq[person_id],
        'file_name': f"{base_name}.csv",
        'num_frames': len(distances[f'Distance_0_1'])
    })
    distances_df.to_csv(output_file, index=False)





def extract_gait_features_from_cycles(input_dir, output_dir):
  
  print("Starting gait feature extraction...")
  print("input directory: ", input_dir, ", output directory: ", output_dir)

  for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.csv') and filename != 'metadata.csv':
        input_file = os.path.join(input_dir, filename)
        extract_gait_features(input_file, output_dir)

  # Save metadata
  metadata_df = pd.DataFrame(metadata_list)
  metadata_df.to_csv(output_dir + '/metadata.csv', index=False)

  print("gait feature extraction completed...")
  return output_dir


