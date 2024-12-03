import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, argrelextrema


metadata_list = []
person_seq = {}

def extract_gait_cycles(input_file, output_dir):
    """
    Extracts gait cycles from a CSV file and saves them as separate CSV files.
    
    Parameters:
    input_file (str): Path to the input CSV file.
    output_dir (str): Path to the directory where the extracted gait cycle CSV files will be saved.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the input CSV file
    df = pd.read_csv(input_file)

    left_ankle = 'joint_17_'
    right_ankle = 'joint_16_'
    # Calculate the Euclidean distance between left and right heel
    df['Ankle_Distance'] = np.sqrt((df[left_ankle+'x'] - df[right_ankle+'x'])**2 + (df[left_ankle+'y'] - df[right_ankle+'y'])**2 + (df[left_ankle+'z'] - df[right_ankle+'z'])**2)

    # Apply Savgol filter to smooth the distance vector
    df['Ankle_Distance_Smoothed'] = savgol_filter(df['Ankle_Distance'], 11, 3)

    # Find all the local maxima (peaks) in the smoothed distance vector
    peak_indices = argrelextrema(df['Ankle_Distance_Smoothed'].values, np.greater)[0]
    peaks = df.iloc[peak_indices]

    # Discard local peaks with a height less than 0.4 × maximum peak height
    max_peak_height = peaks['Ankle_Distance_Smoothed'].max()
    valid_peaks = peaks[peaks['Ankle_Distance_Smoothed'] >= 0.6 * max_peak_height]
    valid_peak_indices = valid_peaks.index.tolist()

    df.drop(['Ankle_Distance', 'Ankle_Distance_Smoothed'], axis=1)

    # Extract the gait cycles
    gait_cycles = []
    for i in range(0, len(valid_peak_indices),3):
        start = valid_peak_indices[i]
        if i+2 < len(valid_peak_indices):
          end = valid_peak_indices[i+2]
          gait_cycle = df.iloc[start:end+1]
          gait_cycles.append(gait_cycle)
            
        # gait_cycles.append(df.iloc[start:end+1])

    # Save the extracted gait cycles as separate CSV files
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    for i, cycle in enumerate(gait_cycles):
        output_file = os.path.join(output_dir, f"{base_name}_{i+1}.csv")

        # Add to metadata
        person_id = base_name[0:9]
        if person_id not in person_seq:
            person_seq[person_id] = 0

        person_seq[person_id] += 1
        metadata_list.append({
            'person_id': person_id,
            'sequence_id': person_seq[person_id],
            'file_name': f"{base_name}_{i+1}.csv",
            'num_frames': len(cycle)
        })
        cycle.to_csv(output_file, index=False)


def extract_gait_cycles_from_csv(input_file, output_dir):
  for filename in os.listdir(input_dir):
    if filename.endswith('.csv') and filename != 'metadata.csv':
        input_file = os.path.join(input_dir, filename)
        extract_gait_cycles(input_file, output_dir)

  # Save metadata
  metadata_df = pd.DataFrame(metadata_list)
  metadata_df.to_csv(output_dir + '/metadata.csv', index=False)

  return output_dir




# input_dir = '/content/CSVdata'
# output_dir = '/content/gait_cycles'
# extract_gait_cycles_from_csv(input_dir,output_dir)