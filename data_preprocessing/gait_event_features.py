import pickle
import shutil
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

metadata_list = []
person_seq = {}
all_features = {}


def extract_gait_events(df, file_path):
    percentages = np.array([(0, 10), (10, 30), (30, 50), (50, 60), (60, 73), (73, 87), (87, 100)])
    total_rows = len(df)

    # Convert DataFrame to NumPy array at the start for faster operations
    data = df.to_numpy()

    # Precompute row indices
    group_indices = (percentages / 100 * total_rows).astype(int)

    # Use numpy to calculate means and stds directly
    means = np.array([data[start:end].mean(axis=0) for start, end in group_indices])
    stds = np.array([data[start:end].std(axis=0) for start, end in group_indices])

    # Concatenate means and stds for fast DataFrame creation
    combined = np.hstack((means, stds))
    
    # Create DataFrame once
    column_names = [f"{col}_mean" for col in df.columns] + [f"{col}_std" for col in df.columns]
    results_df = pd.DataFrame(combined, columns=column_names)

    # Add metadata
    person_id = file_path[:9]
    person_seq[person_id] = person_seq.get(person_id, 0) + 1
    
    metadata_list.append({
        'person_id': person_id,
        'sequence_id': person_seq[person_id],
        'file_name': f"{file_path}.csv",
        'num_frames': len(percentages)
    })

    all_features[f"{file_path}.csv"] = results_df


def extract_gait_events_and_features_from_cycles(input_dir, output_dir):
  
    print("\n\nStarting gait feature extraction from gait events...")
    print("input directory: ", input_dir, ", output directory: ", output_dir)


    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(input_dir + '/data.pkl', 'rb') as f:
        data = pickle.load(f)




    for filename, df in tqdm(data.items()):
        if filename.endswith('.csv') and filename != 'metadata.csv':
            extract_gait_events(df, filename[:-4])


    # shutil.rmtree(input_dir)
    os.remove(input_dir + '/data.pkl')
    with open(output_dir + '/data.pkl', 'wb') as f:
        pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(output_dir + '/metadata.csv', index=False)
    print("gait event feature extraction from gait events completed...\n\n")
    return output_dir
