import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

from .dataset_from_csv import SkeletonDatasetFromCSV

def create_fixed_splits(data_dir, is_skeleton = True, batch_size=32, seed=42, return_dataset = False):
    """
    Create train, validation, and test splits from processed CSV data
    with fixed allocation: 3 for training, 1 for validation, 1 for testing.
    Sequences are randomly shuffled before splitting.

    Args:
        data_dir (str): Directory containing processed CSV files
        batch_size (int): Batch size for data loaders
        seed (int): Random seed for reproducibility

    Returns:
        tuple: Train, validation, and test data loaders
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Read metadata
    metadata = pd.read_csv(Path(data_dir) / 'metadata.csv')

    # Group by person_id
    person_groups = metadata.groupby('person_id')

    train_rows = []
    val_rows = []
    test_rows = []

    for person_id, group in person_groups:
        # Get all sequences for this person and shuffle them
        sequences = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n_sequences = len(sequences)

        n_train = max(1, int(n_sequences * 0.6))
        n_test = max(1, int(n_sequences * 0.2))

        train_rows.append(sequences.iloc[:n_train])
        if n_sequences > n_train:
            test_rows.append(sequences.iloc[n_train:n_train+n_test])
        if n_sequences > n_train + n_test:
            val_rows.append(sequences.iloc[n_train+n_test:])

    # Concatenate all rows
    train_metadata = pd.concat(train_rows, ignore_index=True)
    val_metadata = pd.concat(val_rows, ignore_index=True)
    test_metadata = pd.concat(test_rows, ignore_index=True)

    # Shuffle the final datasets again
    train_metadata = train_metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_metadata = val_metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_metadata = test_metadata.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save split metadata
    train_metadata.to_csv(Path(data_dir) / 'train_metadata.csv', index=False)
    val_metadata.to_csv(Path(data_dir) / 'val_metadata.csv', index=False)
    test_metadata.to_csv(Path(data_dir) / 'test_metadata.csv', index=False)

    k_fold_combined_train_val_metadta = pd.concat([train_metadata, val_metadata], ignore_index=True)
    k_fold_combined_train_val_metadta.to_csv(Path(data_dir) / 'k_fold_combined_train_val_metadata.csv', index=False)

    # Print distribution statistics
    print("\nData split statistics:")
    print(f"Total number of people: {len(person_groups)}")
    print(f"Training samples: {len(train_metadata)}")
    print(f"Validation samples: {len(val_metadata)}")
    print(f"Test samples: {len(test_metadata)}")

    with open(data_dir + '/data.pkl', 'rb') as f:
      data = pickle.load(f)
      print('Loaded data')

    # Create datasets
    if return_dataset:

        train_val_dataset = SkeletonDatasetFromCSV(data, data_dir, 'k_fold_combined_train_val_metadata.csv', is_Skeleton = is_skeleton)
        test_dataset = SkeletonDatasetFromCSV(data, data_dir, 'test_metadata.csv', is_Skeleton = is_skeleton)
        
        train_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader, train_val_dataset, test_dataset
    else:

        train_dataset = SkeletonDatasetFromCSV(data, data_dir, 'train_metadata.csv', is_Skeleton = is_skeleton)
        val_dataset = SkeletonDatasetFromCSV(data, data_dir, 'val_metadata.csv', is_Skeleton = is_skeleton)
        test_dataset = SkeletonDatasetFromCSV(data, data_dir, 'test_metadata.csv', is_Skeleton = is_skeleton)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader

    # return train_loader, val_loader, test_loader



# # Create data loaders and print sample batch
def create_data_loaders(training_data_dir, base_data_dir ,is_skeleton = True, batch_size=32, return_dataset = False):
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_fixed_splits(data_dir=training_data_dir, is_skeleton =  is_skeleton, batch_size=batch_size, return_dataset = return_dataset)

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

        pd.DataFrame([stats]).to_csv(base_data_dir + '/training_dataset_statistics.csv', index=False)
        print("\n\nCreated data loaders....\n\n")
        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

    