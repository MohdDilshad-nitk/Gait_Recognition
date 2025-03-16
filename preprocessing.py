import numpy as np
import pandas as pd
import os
import torch

from data_preprocessing.kgdb_to_csv import process_skeleton_data
from data_preprocessing.data_augmentation import augment_skeleton_data
from data_preprocessing.gait_cycle_extraction import extract_gait_cycles_from_csv, extract_gait_cycles_from_csv_iigc
from data_preprocessing.gait_features_from_cycle import extract_gait_features_from_cycles
from data_preprocessing.gait_event_features import extract_gait_events_and_features_from_cycles


def preprocessor(config):

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)


    # Directory containing the dataset
    base_dir = config['base_dir'] 
    base_data_dir = base_dir + '/data'
    raw_data_dir = base_data_dir + '/Data'
    csv_data_dir = base_data_dir + '/CSVData'
    augmented_data_dir = base_data_dir + '/AugmentedData'
    gait_cycles_dir = base_data_dir + '/GaitCycles'
    gait_features_dir = base_data_dir + '/GaitFeatures'
    gait_event_features_dir = base_data_dir + '/Gait_Event_Features'

    trained_models_dir = base_dir + '/trained_models'
    os.makedirs(trained_models_dir, exist_ok=True)

    drive_checkpoint_dir = config['drive_checkpoint_path']
    os.makedirs(drive_checkpoint_dir, exist_ok=True)


    preprocessing = config['preprocess']

    preprocessing_funcs = {
        'transform': process_skeleton_data,
        'augment': augment_skeleton_data,
        'gait_cycles': extract_gait_cycles_from_csv,
        'gait_cycles_iigc': extract_gait_cycles_from_csv_iigc,
        'gait_features': extract_gait_features_from_cycles,
        'event_features': extract_gait_events_and_features_from_cycles
    }

    output_dirs = {
        'transform': csv_data_dir,
        'augment': augmented_data_dir,
        'gait_cycles': gait_cycles_dir,
        'gait_cycles_iigc': gait_cycles_dir,
        'gait_features': gait_features_dir,
        'event_features': gait_event_features_dir
    }

    last_preprocessing = config.get('last_preprocessing', 'raw')
    prev_dir = output_dirs.get(last_preprocessing, raw_data_dir)
    # prev_dir = raw_data_dir
    for func in preprocessing:
        prev_dir = preprocessing_funcs[func](prev_dir,output_dirs[func])

    return prev_dir

if __name__=="__main__":
    from config import config
    preprocessor(config)