"""
Train a syllable counter model with a balanced dataset.
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

# Import the rule-based syllable counter for validation
from validate_dataset import rule_based_syllable_count


def load_and_balance_dataset(file_path, max_syllables=9, balance_method='undersample', validate=True, validation_threshold=1):
    """
    Load the dataset and balance it to have similar numbers of examples for each syllable count.

    Args:
        file_path: Path to the syllable dictionary CSV file
        max_syllables: Maximum number of syllables to include
        balance_method: Method to balance the dataset ('undersample', 'oversample', or 'weighted')
        validate: Whether to validate the dataset using a rule-based syllable counter
        validation_threshold: Maximum allowed difference between dataset and rule-based counts

    Returns:
        Balanced DataFrame
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Clean the data
    df = df.dropna(subset=['word', 'syl'])
    df['word'] = df['word'].astype(str)

    # Validate the dataset if requested
    if validate:
        print("Validating dataset with rule-based syllable counter...")
        # Add rule-based syllable count
        df['rule_based_syl'] = df['word'].apply(rule_based_syllable_count)

        # Calculate difference
        df['diff'] = df['syl'] - df['rule_based_syl']
        df['abs_diff'] = df['diff'].abs()

        # Filter out words with large differences
        original_size = len(df)
        df = df[df['abs_diff'] <= validation_threshold]
        print(f"Removed {original_size - len(df)} words with abs_diff > {validation_threshold} ({(original_size - len(df))/original_size:.2%} of total)")

        # Print some statistics about the validation
        print(f"Mean absolute difference: {df['abs_diff'].mean():.2f}")
        print(f"Median absolute difference: {df['abs_diff'].median():.2f}")
        print(f"Max absolute difference: {df['abs_diff'].max():.2f}")

        # Print examples of words with different counts
        if len(df[df['abs_diff'] > 0]) > 0:
            print("\nExamples of words with different counts:")
            for _, row in df[df['abs_diff'] > 0].sample(min(10, len(df[df['abs_diff'] > 0]))).iterrows():
                print(f"Word: {row['word']}, Dataset: {row['syl']}, Rule-based: {row['rule_based_syl']}, Diff: {row['diff']}")

    # Filter out words with too many syllables
    df = df[df['syl'] <= max_syllables]

    # Print original distribution
    print("\nOriginal syllable count distribution:")
    for syl, count in df['syl'].value_counts().sort_index().items():
        print(f"{syl} syllables: {count} words ({count/len(df):.2%})")

    # Balance the dataset
    if balance_method == 'undersample':
        # Find the minimum count (excluding very rare counts)
        counts = df['syl'].value_counts()
        min_count = min(counts[counts >= 100])

        # Undersample each class to have the same number of examples
        balanced_df = pd.DataFrame()
        for syl in range(1, max_syllables + 1):
            syl_df = df[df['syl'] == syl]
            if len(syl_df) > min_count:
                syl_df = syl_df.sample(min_count, random_state=42)
            balanced_df = pd.concat([balanced_df, syl_df])

    elif balance_method == 'oversample':
        # Find the maximum count
        max_count = df['syl'].value_counts().max()

        # Oversample each class to have the same number of examples
        balanced_df = pd.DataFrame()
        for syl in range(1, max_syllables + 1):
            syl_df = df[df['syl'] == syl]
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
    for syl, count in balanced_df['syl'].value_counts().sort_index().items():
        print(f"{syl} syllables: {count} words ({count/len(balanced_df):.2%})")

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
    validate=True,
    validation_threshold=1
):
    """
    Train the syllable counter model with a balanced dataset.

    Args:
        data_path: Path to the syllable dictionary CSV file
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
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Load and balance dataset
    df = load_and_balance_dataset(data_path, max_syllables, balance_method, validate, validation_threshold)

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
            }, os.path.join(model_dir, 'syllable_counter_balanced.pt'))

            # Save vocabulary
            with open(os.path.join(model_dir, 'char_vocab_balanced.json'), 'w') as f:
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

    with open(os.path.join(model_dir, 'model_metadata_balanced.json'), 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train syllable counter model with balanced dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/syllable_dictionary.csv",
        help="Path to the syllable dictionary CSV file"
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
        "--validate",
        action="store_true",
        help="Validate the dataset using a rule-based syllable counter"
    )
    parser.add_argument(
        "--validation_threshold",
        type=int,
        default=1,
        help="Maximum allowed difference between dataset and rule-based counts"
    )
    parser.add_argument(
        "--curated_dataset",
        action="store_true",
        help="Use a curated dataset with manually verified syllable counts"
    )

    args = parser.parse_args()

    # If using a curated dataset, check if it exists, otherwise create it
    if args.curated_dataset:
        curated_path = args.data_path.replace('.csv', '_curated.csv')
        if not os.path.exists(curated_path):
            print(f"Curated dataset {curated_path} not found. Creating it...")
            from validate_dataset import validate_dataset, create_curated_dataset

            # Define manual corrections for known problematic words
            manual_corrections = {
                'winebarger': 3,
                'nickelodeon': 5,
                'collaborationist': 7,
                'authoritarianism': 8,
                'supercalifragilistic': 9,  # Should be 9, not 8
                'internationalization': 9,  # Should be 9, not 8
                'institutionalization': 9,  # Should be 9, not 8
                'counterrevolutionary': 8,  # Should be 8, not 9
                'deinstitutionalization': 10,  # Should be 10, not 9
                'extraterritoriality': 10,  # Should be 10, not 9
                'antidisestablishmentarianism': 12  # Should be 12, not 11
            }

            # Validate the dataset
            validated_path = args.data_path.replace('.csv', '_validated.csv')
            validate_dataset(args.data_path, validated_path, 1000, args.validation_threshold)

            # Create the curated dataset
            create_curated_dataset(validated_path, curated_path, manual_corrections)

        # Use the curated dataset
        args.data_path = curated_path
        print(f"Using curated dataset: {args.data_path}")

        # No need to validate again if using a curated dataset
        args.validate = False

    train_model(**vars(args))
