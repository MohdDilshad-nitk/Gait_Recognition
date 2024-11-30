from .model import SkeletonTransformer
import torch
from torch.optim import Adam
import torch.nn as nn
import os
from datetime import datetime
from typing import Dict, Tuple

class SkeletonTransformerTrainer:
    def __init__(
        self,
        model: SkeletonTransformer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
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

            # Update progress bar with current accuracy
            current_accuracy = correct / max(total, 1)
            # val_pbar.set_postfix({'accuracy': f'{current_accuracy:.4f}'})

        return {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'val_accuracy': correct / max(total, 1)
        }

    def augment_sequence(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal augmentations to sequence"""
        # First apply random crop
        sequence, attention_mask = self.augmenter.random_temporal_crop(
            sequence, attention_mask
        )

        # Then apply random masking
        sequence, attention_mask = self.augmenter.random_temporal_mask(
            sequence, attention_mask
        )

        return sequence, attention_mask

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
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

        checkpoint_path = os.path.join(
            self.save_dir,
            f'checkpoint_epoch_{epoch}_{timestamp}.pt'
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

        # Add progress bar for training batches
        # train_pbar = tqdm(self.train_loader, desc='Training', leave=False)

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

            # Update progress bar with current loss and accuracy
            current_accuracy = correct / max(total, 1)
            # train_pbar.set_postfix({
            #     'loss': f'{cls_loss.item():.4f}',
            #     'accuracy': f'{current_accuracy:.4f}'
            # })

        return {
            'train_cls_loss': total_cls_loss / len(self.train_loader),
            'train_accuracy': correct / total
        }

    def train(self, num_epochs: int, resume_path: str = None):
        start_epoch = 0
        if resume_path is not None:
            start_epoch = self.load_checkpoint(resume_path)
            print(f"Resumed training from epoch {start_epoch}")

        # Add progress bar for epochs
        # epoch_pbar = tqdm(range(start_epoch, num_epochs), desc='Training Progress', position=0)

        for epoch in range(start_epoch, num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}

            # Update epoch progress bar with current metrics
            # epoch_pbar.set_postfix({
            #     'train_acc': f"{train_metrics['train_accuracy']:.4f}",
            #     'val_acc': f"{val_metrics['val_accuracy']:.4f}"
            # })

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
                print(f"{k}: {v:.4f}")
            if is_best:
                print("New best model!")
            print()