import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, argrelextrema
from scipy.ndimage import gaussian_filter
from pathlib import Path
from tqdm import tqdm

metadata_list = []
person_seq = {}

def extract_gait_cycles(input_file, output_dir, threshold_fraction = 0.4):
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

    # Apply filter to smooth the distance vector
    # df['Ankle_Distance_Smoothed'] = savgol_filter(df['Ankle_Distance'], 11, 3)
    df['Ankle_Distance_Smoothed'] = gaussian_filter(df['Ankle_Distance'], sigma=5)

    # Find all the local maxima (peaks) in the smoothed distance vector
    peak_indices = argrelextrema(df['Ankle_Distance_Smoothed'].values, np.greater)[0]
    peaks = df.iloc[peak_indices]

    # Discard local peaks with a height less than 0.4 × maximum peak height
    max_peak_height = peaks['Ankle_Distance'].max()

    threshold = threshold_fraction * max_peak_height  # Calculate the threshold
    is_Prev_Peak_Included = True

    peak_groups = []
    peak_group = []

    for index in peak_indices:  # Iterate through each row in the DataFrame
        row = df.iloc[index]  # Get the current row
        if row['Ankle_Distance'] >= threshold:  # Check the condition            
            if is_Prev_Peak_Included:
              peak_group.append(index)
            else:
              if len(peak_group) != 0 :
                peak_groups.append(peak_group)
              peak_group = [index]
            
            is_Prev_Peak_Included = True
        else:
            is_Prev_Peak_Included = False

    if len(peak_group) != 0 :
      peak_groups.append(peak_group)


    #gait cycles
    gait_cycles = []
    df = df.drop(columns=['Ankle_Distance', 'Ankle_Distance_Smoothed'])

    for peak_group in peak_groups:
      #take a window of 3 in the peak group, add the gait cycles, slide the window
      for i in range(0, len(peak_group)):
        start = peak_group[i]
        if i+2 < len(peak_group):
          end = peak_group[i+2]
          gait_cycle = df.iloc[start:end+1]
          gait_cycles.append(gait_cycle)

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



# input_dir = '/content/CSVdata'
# output_dir = '/content/gait_cycles'

def extract_gait_cycles_from_csv(input_dir, output_dir):
  for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.csv') and filename != 'metadata.csv':
        input_file = os.path.join(input_dir, filename)
        extract_gait_cycles(input_file, output_dir,0.4)

  # Save metadata
  metadata_df = pd.DataFrame(metadata_list)
  metadata_df.to_csv(output_dir + '/metadata.csv', index=False)

  print("gait cycle extraction completed...")

  return output_dir

# extract_gait_cycles_from_csv(input_dir,output_dir)
