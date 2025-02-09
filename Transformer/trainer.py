from .model import SkeletonTransformer
import torch
from torch.optim import Adam
import torch.nn as nn
import os
from datetime import datetime
from typing import Dict, Tuple
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class SkeletonTransformerTrainer:
    def __init__(
        self,
        model: SkeletonTransformer,
        train_loader: torch.utils.data.DataLoader | None,
        val_loader: torch.utils.data.DataLoader | None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        save_dir: str = 'models'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.classification_loss = nn.CrossEntropyLoss()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize best metrics for model saving
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        # Add progress bar for validation
        # val_pbar = tqdm(self.val_loader, desc='Validating', leave=False)

        for batch in self.val_loader:
            sequence = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            person_id = batch['person_id'].to(self.device)

            logits = self.model(sequence, attention_mask)
            loss = self.classification_loss(logits, person_id)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == person_id).sum().item()
            total += person_id.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(person_id.cpu().numpy())

            # Update progress bar with current accuracy
            # current_accuracy = correct / max(total, 1)
            # val_pbar.set_postfix({'accuracy': f'{current_accuracy:.4f}'})

        # Compute precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'val_accuracy': correct / max(total, 1),
            'val_prec_recall_f1': (precision, recall, f1)
            
        }

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        fold: int = -1
    ):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch
        }

        if fold != -1:
          checkpoint['fold'] = fold


        # Epochs are 0 indexed but we are displaying everywhere 1 indexed
        path = f'checkpoint_epoch_{epoch+1}_{timestamp}.pt'
        if fold != -1:
          path = f'checkpoint_fold_{fold}_epoch_{epoch+1}_{timestamp}.pt'

        checkpoint_path = os.path.join(
            self.save_dir,
            path
        )

        
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

        if epoch == 0:
            config = {
                'd_model': self.model.d_model,
                'num_encoder_layers': len(self.model.transformer_encoder.layers),
                'dim_feedforward': self.model.transformer_encoder.layers[0].linear1.out_features,
            }
            config_path = os.path.join(self.save_dir, 'model_config.pt')
            torch.save(config, config_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_epoch = checkpoint['best_epoch']

        return checkpoint['epoch']

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_cls_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for batch in self.train_loader:
            sequence = batch['sequence'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            person_id = batch['person_id'].to(self.device)

            logits1 = self.model(sequence, attention_mask)
            cls_loss = self.classification_loss(logits1, person_id)
            loss = cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_cls_loss += cls_loss.item()
            pred = logits1.argmax(dim=1)
            correct += (pred == person_id).sum().item()
            total += person_id.size(0)


            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(person_id.cpu().numpy())

         # Compute precision, recall, and F1-score
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'train_cls_loss': total_cls_loss / len(self.train_loader),
            'train_accuracy': correct / total,
            'train_prec_recall_f1': (precision, recall, f1)
        }

    def train(self, num_epochs: int, resume_path: str = None):
        start_epoch = 0
        if resume_path is not None:
            start_epoch = self.load_checkpoint(resume_path) + 1
            print(f"Resumed training from epoch {start_epoch}")


        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}

            # Check if this is the best model
            is_best = False
            if val_metrics['val_accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['val_accuracy']
                self.best_epoch = epoch
                is_best = True

            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)

            # Print detailed metrics every epoch
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            for k, v in metrics.items():
                if isinstance(v, tuple):
                    formatted_values = tuple(f"{x:.4f}" for x in v)
                    print(f"{k}: {formatted_values}")
                else:
                    print(f"{k}: {v:.4f}")
            if is_best:
                print("New best model!")
            print()

    def train_k_fold(self, num_epochs: int, k_folds: int, dataset, resume_path: str = None):
        """
        Train the model using K-Fold Cross Validation with the ability to resume from a checkpoint.

        Args:
            num_epochs (int): Number of epochs per fold.
            k_folds (int): Number of folds.
            dataset (torch.utils.data.Dataset): Full dataset to be split into K folds.
            resume_path (str, optional): Path to a checkpoint to resume training from.
        """
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_accuracies = []
        start_fold = 0
        start_epoch = 0

        if resume_path is not None:
            checkpoint = torch.load(resume_path)
            start_fold = checkpoint.get('fold', 0)
            start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
            self.load_checkpoint(resume_path)
            print(f"Resumed training from Fold {start_fold+1}, Epoch {start_epoch + 1}") # Epochs are 0-indexed but we are displaying 1 indexed

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            if fold < start_fold:
                continue  # Skip folds that were already trained
            
            print(f"\n--- Training Fold {fold + 1}/{k_folds} ---\n")
            
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32)
            
            self.best_val_accuracy = 0.0  # Reset best accuracy for each fold
            
            for epoch in range(start_epoch if fold == start_fold else 0, num_epochs):
                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                
                metrics = {**train_metrics, **val_metrics}
                
                is_best = False
                if val_metrics['val_accuracy'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['val_accuracy']
                    self.best_epoch = epoch
                    is_best = True
                
                self.save_checkpoint(epoch, metrics, is_best, fold=fold)
                
                print(f"\nFold {fold + 1}, Epoch {epoch+1}/{num_epochs}")

                for k, v in metrics.items():
                    if isinstance(v, tuple):
                        formatted_values = tuple(f"{x:.4f}" for x in v)
                        print(f"{k}: {formatted_values}")
                    else:
                        print(f"{k}: {v:.4f}")
                if is_best:
                    print("New best model for this fold!")
            
            fold_accuracies.append(self.best_val_accuracy)
            start_epoch = 0  # Reset epoch counter for next fold

        print("\nCross-validation results:")
        for i, acc in enumerate(fold_accuracies):
            print(f"Fold {i+1}: Accuracy = {acc:.4f}")

        print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
