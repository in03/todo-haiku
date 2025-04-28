"""
Evaluate the syllable counter model.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from syllable_counter.preprocessing import (
    load_cmudict,
    create_dataset,
    prepare_features
)
from syllable_counter.training import (
    SyllableCounter,
    WordDataset
)


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str = None,
    batch_size: int = 64,
    device: str = None
):
    """
    Evaluate the syllable counter model.

    Args:
        model_path: Path to the trained PyTorch model
        data_path: Path to the CMUdict file
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

    # Load model checkpoint
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

    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(model_path), 'char_vocab.pt')
    char_to_idx = torch.load(vocab_path)

    # Load data
    print("Loading CMUdict...")
    word_to_syllables = load_cmudict(data_path)
    print(f"Loaded {len(word_to_syllables)} words with syllable counts")

    # Create datasets
    print("Creating datasets...")
    _, test_df = create_dataset(word_to_syllables, test_size=0.2)

    # Prepare features
    print("Preparing features...")
    test_features_df = prepare_features(test_df['word'].tolist())

    # Create dataset
    test_dataset = WordDataset(
        test_df['word'].tolist(),
        test_df['syllables'].tolist(),
        test_features_df.drop('word', axis=1).values,
        char_to_idx
    )

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # Evaluate model
    print("Evaluating model...")
    all_predictions = []
    all_labels = []
    all_words = []

    with torch.no_grad():
        for char_ids, features, labels in tqdm(test_loader):
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

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Find misclassified words
    misclassified = []
    for word, true_label, pred_label in zip(all_words, all_labels, all_predictions):
        if true_label != pred_label:
            misclassified.append({
                'word': word,
                'true_syllables': true_label,
                'predicted_syllables': pred_label
            })

    # Print some misclassified examples
    print("\nSample of misclassified words:")
    for i, example in enumerate(misclassified[:20]):
        print(f"{i+1}. '{example['word']}': "
              f"True: {example['true_syllables']}, "
              f"Predicted: {example['predicted_syllables']}")

    # Save results if output directory is specified
    if output_dir:
        # Save misclassified words
        misclassified_df = pd.DataFrame(misclassified)
        misclassified_path = os.path.join(output_dir, 'misclassified.csv')
        misclassified_df.to_csv(misclassified_path, index=False)
        print(f"Saved misclassified words to {misclassified_path}")

        # Save confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.csv')
        pd.DataFrame(cm).to_csv(cm_path, index=False)
        print(f"Saved confusion matrix to {cm_path}")

        # Save evaluation metrics
        metrics = {
            'accuracy': accuracy,
            'num_test_samples': len(all_labels),
            'num_misclassified': len(misclassified)
        }
        metrics_path = os.path.join(output_dir, 'metrics.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate syllable counter model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/syllable_counter.pt",
        help="Path to the trained PyTorch model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/cmudict.dict",
        help="Path to the CMUdict file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/evaluation",
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
