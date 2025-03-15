import numpy as np
import pandas as pd
import os
import torch

# from Transformer.contrastive_trainer import ContSkeletonTransformerTrainer
# from data_loaders.dataset_from_csv import SkeletonDatasetFromCSV
from data_preprocessing.kgdb_to_csv import process_skeleton_data
from data_preprocessing.data_augmentation import augment_skeleton_data
# from data_loaders.train_test_val_loader import create_data_loaders
# from Transformer.model import SkeletonTransformer
# from Transformer.trainer import SkeletonTransformerTrainer
# from Transformer.evaluater import evaluate_model, print_evaluation_results, plot_confusion_matrix
from data_preprocessing.gait_cycle_extraction import extract_gait_cycles_from_csv, extract_gait_cycles_from_csv_iigc
from data_preprocessing.gait_features_from_cycle import extract_gait_features_from_cycles
from data_preprocessing.gait_event_features import extract_gait_events_and_features_from_cycles


# from config import config
# import glob
# from datetime import datetime

#TODO: Check and update max_len in model.py, check what are the maximum number of frames in the dataset, generally for gait cycle its very less like around 30-40, so having maxlen as 5000 is unnecessary
# also for normal case the max number of frames is i guess around 1000(check onnce) so we can set max_len to 1000-1500 try to keep it a power of 2

def preprocessor(config):

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)


    # Directory containing the dataset
    base_dir = config['base_dir'] #'/kaggle/working/Gait_Recognition-main'
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


    # preprocessing
    prev_dir = raw_data_dir
    for func in preprocessing:
        prev_dir = preprocessing_funcs[func](prev_dir,output_dirs[func])

    return prev_dir

if __name__=="__main__":
    from config import config
    preprocessor(config)