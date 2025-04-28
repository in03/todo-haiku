"""
Training script for syllable counter model using the Kaggle dataset.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from syllable_counter.preprocessing import (
    load_kaggle_dataset,
    create_dataset,
    prepare_features
)
from syllable_counter.training import (
    SyllableCounter,
    WordDataset,
    create_char_vocab
)


def train(
    data_path: str,
    model_dir: str,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 0.001,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    device: str = None
):
    """
    Train the syllable counter model using the Kaggle dataset.

    Args:
        data_path: Path to the Kaggle dataset CSV file
        model_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden states
        device: Device to train on ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading Kaggle syllable dictionary...")
    word_to_syllables = load_kaggle_dataset(data_path)
    print(f"Loaded {len(word_to_syllables)} words with syllable counts")

    # Create datasets
    print("Creating datasets...")
    train_df, test_df = create_dataset(word_to_syllables)

    # Prepare features
    print("Preparing features...")
    train_features_df = prepare_features(train_df['word'].tolist())
    test_features_df = prepare_features(test_df['word'].tolist())

    # Create character vocabulary
    all_words = train_df['word'].tolist() + test_df['word'].tolist()
    char_to_idx = create_char_vocab(all_words)
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # Save vocabulary
    vocab_path = os.path.join(model_dir, 'char_vocab.pt')
    torch.save(char_to_idx, vocab_path)
    print(f"Saved vocabulary to {vocab_path}")

    # Create datasets
    train_dataset = WordDataset(
        train_df['word'].tolist(),
        train_df['syllables'].tolist(),
        train_features_df.drop('word', axis=1).values,
        char_to_idx
    )

    test_dataset = WordDataset(
        test_df['word'].tolist(),
        test_df['syllables'].tolist(),
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

            try:
                # Forward pass
                outputs = model(char_ids, features)

                # Print shapes and types for debugging
                if epoch == 0 and progress_bar.n == 0:
                    print(f"Outputs shape: {outputs.shape}, type: {outputs.dtype}")
                    print(f"Labels shape: {labels.shape}, type: {labels.dtype}")
                    print(f"Labels min: {labels.min().item()}, max: {labels.max().item()}")

                # Ensure labels are within the valid range (0-9)
                labels = torch.clamp(labels, 0, 9)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Outputs: {outputs.shape if outputs is not None else 'None'}")
                print(f"Labels: {labels.shape if labels is not None else 'None'}")
                raise

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

                try:
                    # Forward pass
                    outputs = model(char_ids, features)

                    # Ensure labels are within the valid range (0-9)
                    labels = torch.clamp(labels, 0, 9)

                    # Calculate loss
                    loss = criterion(outputs, labels)

                    # Statistics
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    print(f"Outputs: {outputs.shape if outputs is not None else 'None'}")
                    print(f"Labels: {labels.shape if labels is not None else 'None'}")
                    raise

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': test_loss / (progress_bar.n + 1),
                    'acc': 100 * test_correct / test_total
                })

        # Print epoch results
        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%")
        print(f"  Test Loss: {test_loss / len(test_loader):.4f}, "
              f"Test Accuracy: {test_accuracy:.2f}%")

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            model_path = os.path.join(model_dir, 'syllable_counter.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_features': num_features
            }, model_path)
            print(f"Saved best model to {model_path} with accuracy {best_accuracy:.2f}%")

    print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    # Get absolute paths for default values
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    default_data_path = os.path.join(project_dir, "data", "syllable_dictionary.csv")
    default_model_dir = os.path.join(project_dir, "models")

    parser = argparse.ArgumentParser(description="Train syllable counter model using Kaggle dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to the Kaggle dataset CSV file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=default_model_dir,
        help="Directory to save the model"
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
        default=20,
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

    args = parser.parse_args()
    train(**vars(args))
