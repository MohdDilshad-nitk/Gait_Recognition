import numpy as np
import pandas as pd
import torch

from Transformer.contrastive_trainer import ContSkeletonTransformerTrainer
from data_loaders.dataset_from_csv import SkeletonDatasetFromCSV
from data_preprocessing.kgdb_to_csv import process_skeleton_data
from data_preprocessing.data_augmentation import augment_skeleton_data
from data_loaders.train_test_val_loader import create_data_loaders
from Transformer.model import SkeletonTransformer
from Transformer.trainer import SkeletonTransformerTrainer
from Transformer.evaluater import evaluate_model, print_evaluation_results, plot_confusion_matrix
from data_preprocessing.gait_cycle_extraction import extract_gait_cycles_from_csv
from data_preprocessing.gait_features_from_cycle import extract_gait_features_from_cycles
from data_preprocessing.gait_event_features import extract_gait_events_and_features_from_cycles


from config import config

#TODO: Check and update max_len in model.py, check what are the maximum number of frames in the dataset, generally for gait cycle its very less like around 30-40, so having maxlen as 5000 is unnecessary
# also for normal case the max number of frames is i guess around 1000(check onnce) so we can set max_len to 1000-1500 try to keep it a power of 2



# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Directory containing the dataset
base_dir = '/content/Code'
base_data_dir = base_dir + '/data'
raw_data_dir = base_data_dir + '/Data'
csv_data_dir = base_data_dir + '/CSVData'
augmented_data_dir = base_data_dir + '/AugmentedData'
gait_cycles_dir = base_data_dir + '/GaitCycles'
gait_features_dir = base_data_dir + '/GaitFeatures'
gait_event_features_dir = base_data_dir + '/Gait_Event_Features'
trained_models_dir = base_data_dir + '/trained_models'


# training_data_dir = csv_data_dir

preprocessing = config['preprocess']

preprocessing_funcs = {
    'transform': process_skeleton_data,
    'augment': augment_skeleton_data,
    'gait_cycles': extract_gait_cycles_from_csv,
    'gait_features': extract_gait_features_from_cycles,
    'event_features': extract_gait_events_and_features_from_cycles
}

output_dirs = {
    'transform': csv_data_dir,
    'augment': augmented_data_dir,
    'gait_cycles': gait_cycles_dir,
    'gait_features': gait_features_dir,
    'event_features': gait_event_features_dir
}


# preprocessing
prev_dir = raw_data_dir
for func in preprocessing:
    prev_dir = preprocessing_funcs[func](prev_dir,output_dirs[func])




is_skeleton = True
if ('gait_features' in config['preprocess']):
    is_skeleton = False


if config['training']['k_fold']:
    # Create datasets
    dataset = SkeletonDatasetFromCSV(prev_dir, f'train_metadata_k_fold.csv', is_Skeleton=is_skeleton)
else:
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(training_data_dir = prev_dir, base_data_dir = base_data_dir, is_skeleton = is_skeleton, batch_size=32)



max_len = 5000
if ('gait_cycles' in config['preprocess']) or ('gait_features' in config['preprocess']):
    max_len = 128
if ('event_features' in config['preprocess']):
    max_len = 8


d_model = 60
if ('gait_features' in config['preprocess']):
    d_model = 56
if ('event_features' in config['preprocess']):
    d_model = 112


rope = config['training']['rope']
heads = config['training']['nhead']
layers = config['training']['num_encoder_layers']

# Create model and trainer
model = SkeletonTransformer(
    num_joints=20,
    d_model=d_model,
    nhead=heads,
    num_encoder_layers=layers,
    dim_feedforward=256,
    dropout=0.2,
    max_len=max_len,
    num_classes=164,
    rope=rope
)

trainer = None

if config['training']['contrastive']:
    trainer = ContSkeletonTransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=trained_models_dir
    )
else:
    trainer = SkeletonTransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=trained_models_dir
    )

print("\n\nCreated model and model trainer...\n\n")

epochs = config['training']['epochs']

if config['training']['k_fold']:
    trainer.train_k_fold(
        num_epochs=epochs,
        k_folds=5,
        dataset= dataset
    )
else:
    trainer.train(
        num_epochs=epochs,
        resume_path=None  # Set to checkpoint path to resume training
    )

# Save best model
torch.save(model.state_dict(),trained_models_dir + f'/best_model_{epochs}.pt')
print("\n\n training completed... \n\n")


# Evaluate model
results = evaluate_model(model, test_loader)
print_evaluation_results(results)
plot_confusion_matrix(results['confusion_matrix'])

# Save confusion matrix
pd.DataFrame(results['confusion_matrix']).to_csv(f'confusion_matrix_{epochs}.csv')

print("\n\n Evaluation completed... \n\n")