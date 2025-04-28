"""
Train a syllable counter model using a dataset created with the CMU Pronouncing Dictionary.
"""
import argparse
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from syllable_counter.preprocessing import prepare_features
from syllable_counter.training import (
    SyllableCounter,
    WordDataset,
    create_char_vocab
)

# Import the CMU syllable counter
import sys
import os
# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try to import from optimized version first, fall back to original if not available
try:
    from optimized_cmu_counter import count_syllables_hybrid, create_cmu_dataset as cmu_create_dataset
    print("Using optimized CMU syllable counter")
except ImportError:
    from cmu_syllable_counter import count_syllables_hybrid, create_cmu_dataset as cmu_create_dataset
    print("Using original CMU syllable counter")


def create_cmu_dataset(output_path, use_existing=True):
    """
    Create a dataset using the CMU Pronouncing Dictionary.

    Args:
        output_path: Path to save the dataset
        use_existing: Whether to use an existing dataset if available

    Returns:
        DataFrame with words and syllable counts
    """
    if use_existing and os.path.exists(output_path):
        print(f"Using existing CMU dataset: {output_path}")
        return pd.read_csv(output_path)

    print("Creating CMU dataset...")

    # Import necessary functions
    from cmu_syllable_counter import download_cmudict

    # Download CMU dict
    download_cmudict()

    # Create CMU dataset
    df = cmu_create_dataset("data/syllable_dictionary.csv", output_path)

    return df


def load_and_balance_dataset(file_path, max_syllables=9, balance_method='undersample'):
    """
    Load the dataset and balance it to have similar numbers of examples for each syllable count.

    Args:
        file_path: Path to the syllable dictionary CSV file
        max_syllables: Maximum number of syllables to include
        balance_method: Method to balance the dataset ('undersample', 'oversample', or 'weighted')

    Returns:
        Balanced DataFrame
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Clean the data
    df = df.dropna(subset=['word', 'syllables'])
    df['word'] = df['word'].astype(str)

    # Filter out words with too many syllables
    df = df[df['syllables'] <= max_syllables]

    # Print original distribution
    print("\nOriginal syllable count distribution:")
    for syl, count in df['syllables'].value_counts().sort_index().items():
        print(f"{syl} syllables: {count} words ({count/len(df):.2%})")

    # Balance the dataset
    if balance_method == 'undersample':
        # Find the minimum count (excluding very rare counts)
        counts = df['syllables'].value_counts()
        min_count = min(counts[counts >= 100])

        # Undersample each class to have the same number of examples
        balanced_df = pd.DataFrame()
        for syl in range(1, max_syllables + 1):
            syl_df = df[df['syllables'] == syl]
            if len(syl_df) > min_count:
                syl_df = syl_df.sample(min_count, random_state=42)
            balanced_df = pd.concat([balanced_df, syl_df])

    elif balance_method == 'oversample':
        # Find the maximum count
        max_count = df['syllables'].value_counts().max()

        # Oversample each class to have the same number of examples
        balanced_df = pd.DataFrame()
        for syl in range(1, max_syllables + 1):
            syl_df = df[df['syllables'] == syl]
            if len(syl_df) < max_count:
                # Oversample with replacement
                syl_df = syl_df.sample(max_count, replace=True, random_state=42)
            else:
                syl_df = syl_df.sample(max_count, random_state=42)
            balanced_df = pd.concat([balanced_df, syl_df])

    else:  # weighted (no resampling, just use sample_weights in training)
        balanced_df = df

    # Print balanced distribution
    print("\nBalanced syllable count distribution:")
    for syl, count in balanced_df['syllables'].value_counts().sort_index().items():
        print(f"{syl} syllables: {count} words ({count/len(balanced_df):.2%})")

    # Rename 'syllables' to 'syl' to match the expected column name
    balanced_df = balanced_df.rename(columns={'syllables': 'syl'})

    return balanced_df


def train_model(
    data_path,
    model_dir,
    balance_method='undersample',
    max_syllables=9,
    batch_size=64,
    epochs=30,
    learning_rate=0.001,
    embedding_dim=64,
    hidden_dim=128,
    device=None,
    early_stopping_patience=5,
    use_existing=True
):
    """
    Train the syllable counter model with a CMU-based dataset.

    Args:
        data_path: Path to the CMU dataset
        model_dir: Directory to save the model
        balance_method: Method to balance the dataset ('undersample', 'oversample', or 'weighted')
        max_syllables: Maximum number of syllables to include
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden states
        device: Device to train on ('cuda' or 'cpu')
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        use_existing: Whether to use an existing CMU dataset if available
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Create or load CMU dataset
    df = create_cmu_dataset(data_path, use_existing)

    # Load and balance dataset
    df = load_and_balance_dataset(data_path, max_syllables, balance_method)

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['syl'])

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Prepare features
    print("Preparing features...")
    train_features_df = prepare_features(train_df['word'].tolist())
    test_features_df = prepare_features(test_df['word'].tolist())

    # Create character vocabulary
    print("Creating character vocabulary...")
    char_to_idx = create_char_vocab(train_df['word'].tolist())
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Create datasets
    train_dataset = WordDataset(
        train_df['word'].tolist(),
        train_df['syl'].tolist(),
        train_features_df.drop('word', axis=1).values,
        char_to_idx
    )

    test_dataset = WordDataset(
        test_df['word'].tolist(),
        test_df['syl'].tolist(),
        test_features_df.drop('word', axis=1).values,
        char_to_idx
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # Create model
    num_features = train_features_df.drop('word', axis=1).shape[1]
    model = SyllableCounter(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_features=num_features
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for char_ids, features, labels in progress_bar:
            char_ids = char_ids.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(char_ids, features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100 * train_correct / train_total
            })

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
            for char_ids, features, labels in progress_bar:
                char_ids = char_ids.to(device)
                features = features.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(char_ids, features)
                loss = criterion(outputs, labels)

                # Statistics
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': test_loss / (progress_bar.n + 1),
                    'acc': 100 * test_correct / test_total
                })

        # Calculate accuracy
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
              f"Test Accuracy: {test_accuracy:.2f}%")

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            patience_counter = 0

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss / len(train_loader),
                'test_loss': test_loss / len(test_loader),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_features': num_features
            }, os.path.join(model_dir, 'syllable_counter_cmu.pt'))

            # Save vocabulary
            with open(os.path.join(model_dir, 'char_vocab_cmu.json'), 'w') as f:
                json.dump(char_to_idx, f)

            print(f"Model saved at epoch {epoch+1} with test accuracy: {test_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch+1}")
                break

    print(f"Training completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch+1}")

    # Save model metadata
    metadata = {
        'max_word_len': train_dataset.max_word_len,
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_features': num_features,
        'test_accuracy': best_accuracy,
        'balance_method': balance_method,
        'max_syllables': max_syllables
    }

    with open(os.path.join(model_dir, 'model_metadata_cmu.json'), 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train syllable counter model with CMU dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/syllable_dictionary_cmu.csv",
        help="Path to the CMU dataset"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--balance_method",
        type=str,
        choices=['undersample', 'oversample', 'weighted'],
        default='undersample',
        help="Method to balance the dataset"
    )
    parser.add_argument(
        "--max_syllables",
        type=int,
        default=9,
        help="Maximum number of syllables to include"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimension of character embeddings"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of LSTM hidden states"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before stopping"
    )
    parser.add_argument(
        "--use_existing",
        action="store_true",
        help="Use existing CMU dataset if available"
    )

    args = parser.parse_args()
    train_model(**vars(args))

