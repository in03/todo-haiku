"""
Evaluate the balanced syllable counter model and compare with the original model.
"""
import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from syllable_counter.preprocessing import prepare_features
from syllable_counter.training import SyllableCounter, WordDataset


def load_model(model_path, device):
    """
    Load a trained model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model and metadata
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    model = SyllableCounter(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_features=checkpoint['num_features']
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def load_vocabulary(vocab_path):
    """
    Load character vocabulary from a JSON file.
    
    Args:
        vocab_path: Path to the vocabulary file
        
    Returns:
        Character vocabulary dictionary
    """
    with open(vocab_path, 'r') as f:
        return json.load(f)


def evaluate_model(
    model_path,
    balanced_model_path,
    data_path,
    output_dir=None,
    batch_size=64,
    device=None
):
    """
    Evaluate the syllable counter models.
    
    Args:
        model_path: Path to the original model checkpoint
        balanced_model_path: Path to the balanced model checkpoint
        data_path: Path to the test data CSV file
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        device: Device to evaluate on ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    original_model, original_metadata = load_model(model_path, device)
    balanced_model, balanced_metadata = load_model(balanced_model_path, device)
    
    # Load vocabularies
    original_vocab_path = os.path.join(os.path.dirname(model_path), 'char_vocab.json')
    balanced_vocab_path = os.path.join(os.path.dirname(balanced_model_path), 'char_vocab_balanced.json')
    
    original_vocab = load_vocabulary(original_vocab_path)
    balanced_vocab = load_vocabulary(balanced_vocab_path)
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['word', 'syl'])
    df['word'] = df['word'].astype(str)
    
    # Create a stratified sample for testing
    test_df = pd.DataFrame()
    for syl in range(1, 10):  # 1-9 syllables
        syl_df = df[df['syl'] == syl]
        if len(syl_df) > 0:
            # Take up to 500 samples for each syllable count
            sample_size = min(500, len(syl_df))
            test_df = pd.concat([test_df, syl_df.sample(sample_size, random_state=42)])
    
    print(f"Test set size: {len(test_df)}")
    print("Syllable distribution in test set:")
    for syl, count in test_df['syl'].value_counts().sort_index().items():
        print(f"{syl} syllables: {count} words ({count/len(test_df):.2%})")
    
    # Prepare features
    print("Preparing features...")
    test_features_df = prepare_features(test_df['word'].tolist())
    
    # Create datasets for both models
    original_dataset = WordDataset(
        test_df['word'].tolist(),
        test_df['syl'].tolist(),
        test_features_df.drop('word', axis=1).values,
        original_vocab
    )
    
    balanced_dataset = WordDataset(
        test_df['word'].tolist(),
        test_df['syl'].tolist(),
        test_features_df.drop('word', axis=1).values,
        balanced_vocab
    )
    
    # Create data loaders
    original_loader = torch.utils.data.DataLoader(
        original_dataset,
        batch_size=batch_size
    )
    
    balanced_loader = torch.utils.data.DataLoader(
        balanced_dataset,
        batch_size=batch_size
    )
    
    # Evaluate models
    print("Evaluating original model...")
    original_predictions, original_labels, original_words = evaluate_single_model(
        original_model, original_loader, test_df, device
    )
    
    print("Evaluating balanced model...")
    balanced_predictions, balanced_labels, balanced_words = evaluate_single_model(
        balanced_model, balanced_loader, test_df, device
    )
    
    # Calculate accuracy
    original_accuracy = accuracy_score(original_labels, original_predictions)
    balanced_accuracy = accuracy_score(balanced_labels, balanced_predictions)
    
    print(f"Original model accuracy: {original_accuracy:.4f}")
    print(f"Balanced model accuracy: {balanced_accuracy:.4f}")
    
    # Calculate confusion matrices
    original_cm = confusion_matrix(original_labels, original_predictions)
    balanced_cm = confusion_matrix(balanced_labels, balanced_predictions)
    
    print("Original model confusion matrix:")
    print(original_cm)
    
    print("Balanced model confusion matrix:")
    print(balanced_cm)
    
    # Calculate classification reports
    original_report = classification_report(original_labels, original_predictions)
    balanced_report = classification_report(balanced_labels, balanced_predictions)
    
    print("Original model classification report:")
    print(original_report)
    
    print("Balanced model classification report:")
    print(balanced_report)
    
    # Find misclassified words for both models
    original_misclassified = []
    for word, true_label, pred_label in zip(original_words, original_labels, original_predictions):
        if true_label != pred_label:
            original_misclassified.append({
                'word': word,
                'true_syllables': true_label,
                'predicted_syllables': pred_label,
                'error': pred_label - true_label
            })
    
    balanced_misclassified = []
    for word, true_label, pred_label in zip(balanced_words, balanced_labels, balanced_predictions):
        if true_label != pred_label:
            balanced_misclassified.append({
                'word': word,
                'true_syllables': true_label,
                'predicted_syllables': pred_label,
                'error': pred_label - true_label
            })
    
    # Calculate error statistics
    original_errors = [m['error'] for m in original_misclassified]
    balanced_errors = [m['error'] for m in balanced_misclassified]
    
    print(f"Original model average error: {np.mean(original_errors):.2f}")
    print(f"Balanced model average error: {np.mean(balanced_errors):.2f}")
    
    print(f"Original model error standard deviation: {np.std(original_errors):.2f}")
    print(f"Balanced model error standard deviation: {np.std(balanced_errors):.2f}")
    
    # Print some misclassified examples
    print("\nSample of original model misclassified words:")
    for i, example in enumerate(sorted(original_misclassified, key=lambda x: abs(x['error']), reverse=True)[:20]):
        print(f"{i+1}. '{example['word']}': "
              f"True: {example['true_syllables']}, "
              f"Predicted: {example['predicted_syllables']}, "
              f"Error: {example['error']}")
    
    print("\nSample of balanced model misclassified words:")
    for i, example in enumerate(sorted(balanced_misclassified, key=lambda x: abs(x['error']), reverse=True)[:20]):
        print(f"{i+1}. '{example['word']}': "
              f"True: {example['true_syllables']}, "
              f"Predicted: {example['predicted_syllables']}, "
              f"Error: {example['error']}")
    
    # Save evaluation results
    if output_dir:
        # Save misclassified words
        pd.DataFrame(original_misclassified).to_csv(
            os.path.join(output_dir, 'original_misclassified.csv'), index=False
        )
        pd.DataFrame(balanced_misclassified).to_csv(
            os.path.join(output_dir, 'balanced_misclassified.csv'), index=False
        )
        
        # Save confusion matrices
        np.save(os.path.join(output_dir, 'original_confusion_matrix.npy'), original_cm)
        np.save(os.path.join(output_dir, 'balanced_confusion_matrix.npy'), balanced_cm)
        
        # Save classification reports
        with open(os.path.join(output_dir, 'original_classification_report.txt'), 'w') as f:
            f.write(original_report)
        with open(os.path.join(output_dir, 'balanced_classification_report.txt'), 'w') as f:
            f.write(balanced_report)
        
        print(f"Evaluation results saved to {output_dir}")


def evaluate_single_model(model, data_loader, test_df, device):
    """
    Evaluate a single model.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        test_df: Test DataFrame
        device: Device to evaluate on
        
    Returns:
        Tuple of (predictions, labels, words)
    """
    all_predictions = []
    all_labels = []
    all_words = []
    
    with torch.no_grad():
        for char_ids, features, labels in tqdm(data_loader):
            char_ids = char_ids.to(device)
            features = features.to(device)
            
            # Forward pass
            outputs = model(char_ids, features)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Get the corresponding words
            batch_indices = range(len(all_words), len(all_words) + len(labels))
            batch_words = [test_df['word'].iloc[i % len(test_df)] for i in batch_indices]
            all_words.extend(batch_words)
    
    return all_predictions, all_labels, all_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate syllable counter models")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/syllable_counter.pt",
        help="Path to the original model checkpoint"
    )
    parser.add_argument(
        "--balanced_model_path",
        type=str,
        default="models/syllable_counter_balanced.pt",
        help="Path to the balanced model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/syllable_dictionary.csv",
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on ('cuda' or 'cpu')"
    )
    
    args = parser.parse_args()
    evaluate_model(**vars(args))
