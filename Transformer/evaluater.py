import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
# import seaborn as sns



# Initialize model and move to device
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = YourModel().to(device)
default_criterion = nn.CrossEntropyLoss()

def evaluate_model(model, test_loader, criterion = default_criterion, device=default_device):
    """
    Evaluate the model on test data

    Args:
        model (nn.Module): The trained model
        test_loader (DataLoader): DataLoader containing test data
        criterion (nn.Module): Loss function
        device (str): Device to run evaluation on ('cuda' or 'cpu')

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Get batch data
            sequences = batch['sequence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            person_ids = batch['person_id'].to(device)

            # Forward pass
            outputs = model(sequences, attention_mask)
            loss = criterion(outputs, person_ids)

            # Calculate predictions
            predictions = torch.argmax(outputs, dim=1)

            # Accumulate results
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(person_ids.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Get detailed classification report
    report = classification_report(all_labels, all_predictions, output_dict=True)

    return {
        'test_loss': avg_loss,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'true_labels': all_labels
    }

def plot_confusion_matrix(confusion_matrix, class_names=None):
    """
    Plot confusion matrix as a heatmap

    Args:
        confusion_matrix (np.ndarray): The confusion matrix to plot
        class_names (list, optional): List of class names
    """
    plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if class_names:
        plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45)
    plt.tight_layout()
    plt.show()

def print_evaluation_results(results):
    """
    Print evaluation metrics in a formatted way

    Args:
        results (dict): Dictionary containing evaluation results
    """
    print("\nEvaluation Results:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")

    print("\nOverall Metrics:")
    macro_avg = results['classification_report']['macro avg']
    print(f"Macro Avg - Precision: {macro_avg['precision']:.3f}, "
          f"Recall: {macro_avg['recall']:.3f}, "
          f"F1-Score: {macro_avg['f1-score']:.3f}")
    
    # print("\nDetailed Classification Report:")
    # # Print per-class metrics
    # for class_id in results['classification_report'].keys():
    #     if class_id not in ['accuracy', 'macro avg', 'weighted avg']:
    #         metrics = results['classification_report'][class_id]
    #         print(f"\nClass {class_id}:")
    #         print(f"Precision: {metrics['precision']:.3f}")
    #         print(f"Recall: {metrics['recall']:.3f}")
    #         print(f"F1-Score: {metrics['f1-score']:.3f}")
    #         print(f"Support: {metrics['support']}")


