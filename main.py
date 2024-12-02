import numpy as np
import pandas as pd
import torch

from data_preprocessing.kgdb_to_csv import process_skeleton_data
from data_preprocessing.data_augmentation import augment_skeleton_data
from data_loaders.train_test_val_loader import create_fixed_splits
from Transformer.model import SkeletonTransformer
from Transformer.trainer import SkeletonTransformerTrainer
from Transformer.evaluater import evaluate_model, print_evaluation_results, plot_confusion_matrix


# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Directory containing the dataset
base_dir = '/content/Code'
base_data_dir = base_dir + '/data'
raw_data_dir = base_data_dir + '/Data'
csv_data_dir = base_data_dir + '/CSVData'
augmented_data_dir = base_data_dir + '/AugmentedData'
trained_models_dir = base_data_dir + '/trained_models'


# Transform data
process_skeleton_data(raw_data_dir, csv_data_dir)
print("Transformation completed...")

# Augment data
augment_skeleton_data(csv_data_dir, augmented_data_dir)
print("augmentation completed...")


# Create data loaders
sb = ''
try:
    # Create data loaders
    train_loader, val_loader, test_loader = create_fixed_splits(data_dir=augmented_data_dir, batch_size=32)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Get a sample batch
    sample_batch = next(iter(train_loader))
    # print(sample_batch)
    print("\nSample batch contents:")
    for key, value in sample_batch.items():
        if torch.is_tensor(value):
            print(f"{key} shape: {value.shape}")
        else:
            print(f"{key}: {value}")

    # Save dataset statistics
    stats = {
        'num_training_sequences': len(train_loader.dataset),
        'num_validation_sequences': len(val_loader.dataset),
        'num_test_sequences': len(test_loader.dataset),
        'max_sequence_length': train_loader.dataset.max_len,
        'num_joints': 20,
        'num_persons': len(train_loader.dataset.person_ids)
    }

    pd.DataFrame([stats]).to_csv(base_data_dir + '/augmented_dataset_statistics.csv', index=False)

except Exception as e:
    print(f"Error processing dataset: {str(e)}")


print("\n\nCreated data loaders....\n\n")



# Create model and trainer
model = SkeletonTransformer(
    num_joints=20,
    d_model=60,
    nhead=1,
    num_encoder_layers=1,
    dim_feedforward=256,
    dropout=0.2,
    num_classes=164,
    rope=True
)

trainer = SkeletonTransformerTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    save_dir=trained_models_dir
)

print("\n\nCreated model and model trainer...\n\n")



# Train and save model
trainer.train(
    num_epochs=60,
    resume_path=None  # Set to checkpoint path to resume training
)

torch.save(model.state_dict(),trained_models_dir + '/best_aug_model_60.pt')
print("\n\n training completed... \n\n")




# Evaluate model
results = evaluate_model(model, test_loader)
print_evaluation_results(results)
plot_confusion_matrix(results['confusion_matrix'])

# Save confusion matrix
pd.DataFrame(results['confusion_matrix']).to_csv('confusion_matrix_60.csv')

print("\n\n Evaluation completed... \n\n")