import numpy as np
import pandas as pd
import torch

from Transformer.contrastive_trainer import ContSkeletonTransformerTrainer
from data_preprocessing.kgdb_to_csv import process_skeleton_data
from data_preprocessing.data_augmentation import augment_skeleton_data
from data_loaders.train_test_val_loader import create_data_loaders
from Transformer.model import SkeletonTransformer
from Transformer.trainer import SkeletonTransformerTrainer
from Transformer.evaluater import evaluate_model, print_evaluation_results, plot_confusion_matrix
from data_preprocessing.gait_cycle_extraction import extract_gait_cycles_from_csv
from data_preprocessing.gait_features_from_cycle import extract_gait_features_from_cycles



#TODO: Check and update max_len in model.py, check what are the maximum number of frames in the dataset, generally for gait cycle its very less like around 30-40, so having maxlen as 5000 is unnecessary
# also for normal case the max number of frames is i guess around 1000(check onnce) so we can set max_len to 1000-1500 try to keep it a power of 2


config = {
    'transform' : True,
    'augment' : True,
    'gait_cycles' : True,
    'gait_features' : True,
    'rope' : True,
    'contrastive' : True,
    'epochs' : 60
}


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
trained_models_dir = base_data_dir + '/trained_models'


training_data_dir = csv_data_dir

# Transform data
if config['transform']:
    training_data_dir = process_skeleton_data(raw_data_dir, csv_data_dir)


# Augment data
if config['augment']:
    training_data_dir = augment_skeleton_data(csv_data_dir, augmented_data_dir)


#Extract gait cycles
if config['gait_cycles']:
    training_data_dir = extract_gait_cycles_from_csv(csv_data_dir,gait_cycles_dir)


#Extract gait features
if config['gait_features']:
    training_data_dir = extract_gait_features_from_cycles(gait_cycles_dir,gait_features_dir)



# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(data_dir=training_data_dir, base_data_dir= base_data_dir, batch_size=32)

# sb = ''
# try:
#     # Create data loaders
#     train_loader, val_loader, test_loader = create_fixed_splits(data_dir=training_data_dir, batch_size=32)

#     print(f"Number of training batches: {len(train_loader)}")
#     print(f"Number of validation batches: {len(val_loader)}")
#     print(f"Number of test batches: {len(test_loader)}")

#     # Get a sample batch
#     sample_batch = next(iter(train_loader))
#     # print(sample_batch)
#     print("\nSample batch contents:")
#     for key, value in sample_batch.items():
#         if torch.is_tensor(value):
#             print(f"{key} shape: {value.shape}")
#         else:
#             print(f"{key}: {value}")

#     # Save dataset statistics
#     stats = {
#         'num_training_sequences': len(train_loader.dataset),
#         'num_validation_sequences': len(val_loader.dataset),
#         'num_test_sequences': len(test_loader.dataset),
#         'max_sequence_length': train_loader.dataset.max_len,
#         'num_joints': 20,
#         'num_persons': len(train_loader.dataset.person_ids)
#     }

#     pd.DataFrame([stats]).to_csv(base_data_dir + '/augmented_dataset_statistics.csv', index=False)

# except Exception as e:
#     print(f"Error processing dataset: {str(e)}")


# print("\n\nCreated data loaders....\n\n")
max_len = 128 if config['gait_cycles'] or config['gait_features'] else 5000
rope = config['rope']


# Create model and trainer
model = SkeletonTransformer(
    num_joints=20,
    d_model=60,
    nhead=1,
    num_encoder_layers=1,
    dim_feedforward=256,
    dropout=0.2,
    max_len=max_len,
    num_classes=164,
    rope=rope
)

trainer = None

if config['contrastive']:
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


trainer.train(
    num_epochs=config['epochs'],
    resume_path=None  # Set to checkpoint path to resume training
)

torch.save(model.state_dict(),trained_models_dir + f'/best_model_{config['epochs']}.pt')
print("\n\n training completed... \n\n")




# Evaluate model
results = evaluate_model(model, test_loader)
print_evaluation_results(results)
plot_confusion_matrix(results['confusion_matrix'])

# Save confusion matrix
pd.DataFrame(results['confusion_matrix']).to_csv(f'confusion_matrix_{config['epochs']}.csv')

print("\n\n Evaluation completed... \n\n")