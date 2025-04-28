"""
Train a character-level LSTM model for syllable detection.
Treats syllabification as a sequence labeling task (BIO tags).
"""
import argparse
import os
import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import the CMU syllable counter
import sys
import os
# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try to import from optimized version first, fall back to original if not available
try:
    from optimized_cmu_counter import count_syllables_cmu, download_cmudict
    print("Using optimized CMU syllable counter")
except ImportError:
    from cmu_syllable_counter import count_syllables_cmu, download_cmudict
    print("Using original CMU syllable counter")


# Character-level LSTM model for syllable detection
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [batch size, seq len]
        embedded = self.embedding(text)
        # embedded = [batch size, seq len, embedding dim]
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), 
                                                          batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output = [batch size, seq len, hidden dim * 2]
        
        output = self.dropout(output)
        
        # Pass through linear layer
        prediction = self.fc(output)
        # prediction = [batch size, seq len, output dim]
        
        return prediction


# Dataset for character-level syllable detection
class SyllableDataset(Dataset):
    def __init__(self, words, labels, char_to_idx):
        self.words = words
        self.labels = labels
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        
        # Convert characters to indices
        char_indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in word]
        
        return {
            'word': word,
            'chars': torch.tensor(char_indices, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'length': len(word)
        }


# Collate function for batching
def collate_fn(batch):
    words = [item['word'] for item in batch]
    chars = [item['chars'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {
        'words': words,
        'chars': chars_padded,
        'labels': labels_padded,
        'lengths': lengths
    }


# Create character vocabulary
def create_char_vocab(words):
    chars = set()
    for word in words:
        chars.update(word.lower())
    
    # Add special tokens
    char_to_idx = {'<pad>': 0, '<unk>': 1}
    for i, char in enumerate(sorted(chars)):
        char_to_idx[char] = i + 2
    
    return char_to_idx


# Get syllable boundaries from CMU dictionary
def get_syllable_boundaries(word):
    """
    Get syllable boundaries for a word using the CMU dictionary.
    Returns a list of BIO tags (B = beginning of syllable, I = inside, O = outside).
    """
    # Get pronunciation from CMU dict
    pronunciation = count_syllables_cmu(word)
    
    if pronunciation is None:
        # Fall back to rule-based approach
        return get_syllable_boundaries_rule_based(word)
    
    # Get the actual pronunciation
    from nltk.corpus import cmudict
    d = cmudict.dict()
    
    if word.lower() not in d:
        return get_syllable_boundaries_rule_based(word)
    
    phones = d[word.lower()][0]  # Use first pronunciation
    
    # Find vowel phonemes (those containing a digit)
    vowel_indices = [i for i, phone in enumerate(phones) if re.search(r'\d', phone)]
    
    # Map phoneme indices to character indices
    # This is an approximation since there's not a 1:1 mapping
    char_indices = []
    char_idx = 0
    for i, phone in enumerate(phones):
        if i in vowel_indices:
            char_indices.append(char_idx)
        char_idx = min(char_idx + 1, len(word) - 1)
    
    # Create BIO tags
    bio_tags = ['I'] * len(word)
    for idx in char_indices:
        bio_tags[idx] = 'B'
    
    # Convert to numeric labels
    # B = 2, I = 1, O = 0
    numeric_labels = [2 if tag == 'B' else 1 if tag == 'I' else 0 for tag in bio_tags]
    
    return numeric_labels


# Rule-based syllable boundary detection
def get_syllable_boundaries_rule_based(word):
    """
    Get syllable boundaries for a word using a rule-based approach.
    Returns a list of BIO tags (B = beginning of syllable, I = inside, O = outside).
    """
    vowels = 'aeiouy'
    bio_tags = ['I'] * len(word)
    
    # First character of a vowel sequence is the beginning of a syllable
    in_vowel = False
    for i, char in enumerate(word.lower()):
        if char in vowels:
            if not in_vowel:
                bio_tags[i] = 'B'
                in_vowel = True
        else:
            in_vowel = False
    
    # If no syllable was found, mark the first character as the beginning
    if 'B' not in bio_tags and len(bio_tags) > 0:
        bio_tags[0] = 'B'
    
    # Convert to numeric labels
    # B = 2, I = 1, O = 0
    numeric_labels = [2 if tag == 'B' else 1 if tag == 'I' else 0 for tag in bio_tags]
    
    return numeric_labels


# Prepare dataset
def prepare_dataset(data_path, max_samples=None):
    """
    Prepare dataset for syllable detection.
    
    Args:
        data_path: Path to the syllable dictionary CSV file
        max_samples: Maximum number of samples to use (for debugging)
        
    Returns:
        words, labels, char_to_idx
    """
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Clean the data
    df = df.dropna(subset=['word'])
    df['word'] = df['word'].astype(str)
    
    # Limit samples if specified
    if max_samples is not None:
        df = df.sample(min(max_samples, len(df)), random_state=42)
    
    # Get words
    words = df['word'].tolist()
    
    # Create character vocabulary
    char_to_idx = create_char_vocab(words)
    
    # Download CMU dict
    download_cmudict()
    
    # Get syllable boundaries
    print("Getting syllable boundaries...")
    import re  # Import here to avoid circular import
    labels = []
    for word in tqdm(words):
        labels.append(get_syllable_boundaries(word))
    
    return words, labels, char_to_idx


# Count syllables from BIO tags
def count_syllables_from_bio(bio_tags):
    """
    Count syllables from BIO tags.
    
    Args:
        bio_tags: List of BIO tags (2 = B, 1 = I, 0 = O)
        
    Returns:
        Number of syllables
    """
    # Count 'B' tags
    return sum(1 for tag in bio_tags if tag == 2)


# Train model
def train_model(
    data_path,
    model_dir,
    batch_size=64,
    epochs=30,
    learning_rate=0.001,
    embedding_dim=64,
    hidden_dim=128,
    n_layers=2,
    dropout=0.2,
    device=None,
    early_stopping_patience=5,
    max_samples=None
):
    """
    Train the character-level LSTM model for syllable detection.
    
    Args:
        data_path: Path to the syllable dictionary CSV file
        model_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden states
        n_layers: Number of LSTM layers
        dropout: Dropout rate
        device: Device to train on ('cuda' or 'cpu')
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        max_samples: Maximum number of samples to use (for debugging)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare dataset
    words, labels, char_to_idx = prepare_dataset(data_path, max_samples)
    
    # Split into train and test sets
    train_words, test_words, train_labels, test_labels = train_test_split(
        words, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(train_words)}")
    print(f"Test set size: {len(test_words)}")
    
    # Create datasets
    train_dataset = SyllableDataset(train_words, train_labels, char_to_idx)
    test_dataset = SyllableDataset(test_words, test_labels, char_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    # Create model
    model = CharLSTM(
        vocab_size=len(char_to_idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=3,  # B, I, O
        n_layers=n_layers,
        dropout=dropout
    ).to(device)
    
    # Define loss function and optimizer
    # Use class weights to handle imbalance
    # B (2) is rare, I (1) is common, O (0) is rare
    class_weights = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # Ignore padding
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
        train_syllable_correct = 0
        train_syllable_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in progress_bar:
            chars = batch['chars'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(chars, lengths)
            
            # Reshape for loss calculation
            batch_size, seq_len, output_dim = outputs.shape
            outputs = outputs.view(-1, output_dim)
            labels = labels.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            
            # Calculate token-level accuracy
            _, predicted = torch.max(outputs, 1)
            mask = labels != 0  # Ignore padding
            train_total += mask.sum().item()
            train_correct += ((predicted == labels) & mask).sum().item()
            
            # Calculate syllable count accuracy
            for i in range(batch_size):
                word_length = lengths[i].item()
                word_labels = labels[i*seq_len:i*seq_len+word_length]
                word_predicted = predicted[i*seq_len:i*seq_len+word_length]
                
                actual_count = count_syllables_from_bio(word_labels)
                predicted_count = count_syllables_from_bio(word_predicted)
                
                train_syllable_total += 1
                if actual_count == predicted_count:
                    train_syllable_correct += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100 * train_correct / max(1, train_total),
                'syl_acc': 100 * train_syllable_correct / max(1, train_syllable_total)
            })
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_syllable_correct = 0
        test_syllable_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
            for batch in progress_bar:
                chars = batch['chars'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths'].to(device)
                
                # Forward pass
                outputs = model(chars, lengths)
                
                # Reshape for loss calculation
                batch_size, seq_len, output_dim = outputs.shape
                outputs = outputs.view(-1, output_dim)
                labels = labels.view(-1)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Statistics
                test_loss += loss.item()
                
                # Calculate token-level accuracy
                _, predicted = torch.max(outputs, 1)
                mask = labels != 0  # Ignore padding
                test_total += mask.sum().item()
                test_correct += ((predicted == labels) & mask).sum().item()
                
                # Calculate syllable count accuracy
                for i in range(batch_size):
                    word_length = lengths[i].item()
                    word_labels = labels[i*seq_len:i*seq_len+word_length]
                    word_predicted = predicted[i*seq_len:i*seq_len+word_length]
                    
                    actual_count = count_syllables_from_bio(word_labels)
                    predicted_count = count_syllables_from_bio(word_predicted)
                    
                    test_syllable_total += 1
                    if actual_count == predicted_count:
                        test_syllable_correct += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': test_loss / (progress_bar.n + 1),
                    'acc': 100 * test_correct / max(1, test_total),
                    'syl_acc': 100 * test_syllable_correct / max(1, test_syllable_total)
                })
        
        # Calculate accuracy
        train_accuracy = 100 * train_correct / max(1, train_total)
        test_accuracy = 100 * test_correct / max(1, test_total)
        train_syllable_accuracy = 100 * train_syllable_correct / max(1, train_syllable_total)
        test_syllable_accuracy = 100 * test_syllable_correct / max(1, test_syllable_total)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Train Syllable Accuracy: {train_syllable_accuracy:.2f}%")
        print(f"Test Loss: {test_loss / len(test_loader):.4f}, "
              f"Test Accuracy: {test_accuracy:.2f}%, "
              f"Test Syllable Accuracy: {test_syllable_accuracy:.2f}%")
        
        # Print examples
        if epoch % 5 == 0:
            print("\nExample predictions:")
            for i in range(min(5, batch_size)):
                word = batch['words'][i]
                word_length = lengths[i].item()
                word_labels = labels[i*seq_len:i*seq_len+word_length]
                word_predicted = predicted[i*seq_len:i*seq_len+word_length]
                
                actual_count = count_syllables_from_bio(word_labels)
                predicted_count = count_syllables_from_bio(word_predicted)
                
                print(f"Word: {word}")
                print(f"Actual: {word_labels.tolist()}, Count: {actual_count}")
                print(f"Predicted: {word_predicted.tolist()}, Count: {predicted_count}")
                print()
        
        # Save the best model
        if test_syllable_accuracy > best_accuracy:
            best_accuracy = test_syllable_accuracy
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
                'train_syllable_accuracy': train_syllable_accuracy,
                'test_syllable_accuracy': test_syllable_accuracy,
                'vocab_size': len(char_to_idx),
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'n_layers': n_layers,
                'dropout': dropout
            }, os.path.join(model_dir, 'syllable_char_lstm.pt'))
            
            # Save vocabulary
            with open(os.path.join(model_dir, 'char_vocab_lstm.json'), 'w') as f:
                json.dump(char_to_idx, f)
            
            print(f"Model saved at epoch {epoch+1} with test syllable accuracy: {test_syllable_accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch+1}")
                break
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}% at epoch {best_epoch+1}")
    
    # Save model metadata
    metadata = {
        'vocab_size': len(char_to_idx),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'dropout': dropout,
        'test_accuracy': best_accuracy
    }
    
    with open(os.path.join(model_dir, 'model_metadata_char_lstm.json'), 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train character-level LSTM model for syllable detection")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/syllable_dictionary_cmu.csv",
        help="Path to the syllable dictionary CSV file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
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
        "--n_layers",
        type=int,
        default=2,
        help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
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
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    
    args = parser.parse_args()
    train_model(**vars(args))
