import pandas as pd
import numpy as np
import os
from pathlib import Path

metadata_list = []
person_seq = {}

# Step 1: Read the CSV file
def extract_gait_events(file_path, output_dir):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Step 2: Define percentage group ranges
    percentages = [(0, 10), (10, 30), (30, 50), (50, 60), (60, 73), (73, 87), (87, 100)]

    # Calculate the row indices for each percentage group
    total_rows = len(df)
    group_indices = [(round(start / 100 * total_rows), round(end / 100 * total_rows)) for start, end in percentages]

    # Step 3: Compute mean and standard deviation for each group
    results = []
    for start_idx, end_idx in group_indices:
        group = df.iloc[start_idx:end_idx]
    
        group_means = group.mean()
        group_stds = group.std()

        # Combine means and stds into a single row
        combined = pd.concat([group_means, group_stds], ignore_index=True, axis=0)
        results.append(combined)

    # Step 4: Create a new DataFrame and save to CSV
    results_df = pd.DataFrame(results)

    # Rename columns to reflect mean and std
    column_names = [f"{col}_mean" for col in df.columns] + [f"{col}_std" for col in df.columns]
    results_df.columns = column_names

    # Save the extracted gait features as separate CSV files
    base_name = os.path.splitext(os.path.basename(file_path))[0]
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
        'num_frames': 7
    })

    # Save the results to a new CSV file
    results_df.to_csv(output_file, index=False)


def extract_gait_events_and_features_from_cycles(input_dir, output_dir):
  for filename in os.listdir(input_dir):
    if filename.endswith('.csv') and filename != 'metadata.csv':
        input_file = os.path.join(input_dir, filename)
        extract_gait_events(input_file, output_dir)

  # Save metadata
  metadata_df = pd.DataFrame(metadata_list)
  metadata_df.to_csv(output_dir + '/metadata.csv', index=False)

  return output_dir
