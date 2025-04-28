"""
Model definition for syllable counting.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CharacterEncoder(nn.Module):
    """
    Character-level encoder for words.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate the final hidden states from both directions
        # hidden shape: (2, batch_size, hidden_dim)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        # hidden shape: (batch_size, 2*hidden_dim)
        return hidden


class SyllableCounter(nn.Module):
    """
    Neural network for syllable counting.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = 64, 
        hidden_dim: int = 128, 
        num_features: int = 7
    ):
        super().__init__()
        self.char_encoder = CharacterEncoder(vocab_size, embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2 + num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)  # Output 0-9 syllables
        
    def forward(self, char_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # char_ids shape: (batch_size, seq_len)
        # features shape: (batch_size, num_features)
        char_encoding = self.char_encoder(char_ids)
        # char_encoding shape: (batch_size, 2*hidden_dim)
        
        # Concatenate character encoding with additional features
        combined = torch.cat((char_encoding, features), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class WordDataset(Dataset):
    """
    Dataset for word syllable counting.
    """
    def __init__(
        self, 
        words: List[str], 
        syllable_counts: List[int], 
        features: np.ndarray, 
        char_to_idx: dict
    ):
        self.words = words
        self.syllable_counts = syllable_counts
        self.features = features
        self.char_to_idx = char_to_idx
        self.max_word_len = max(len(word) for word in words)
        
    def __len__(self) -> int:
        return len(self.words)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        syllables = self.syllable_counts[idx]
        features = self.features[idx]
        
        # Convert word to character indices
        char_ids = [self.char_to_idx.get(c, 0) for c in word]
        # Pad to max length
        char_ids = char_ids + [0] * (self.max_word_len - len(char_ids))
        
        return (
            torch.tensor(char_ids, dtype=torch.long),
            torch.tensor(features, dtype=torch.float),
            torch.tensor(syllables, dtype=torch.long)
        )


def create_char_vocab(words: List[str]) -> dict:
    """
    Create a character vocabulary from a list of words.
    
    Args:
        words: List of words
        
    Returns:
        Dictionary mapping characters to indices
    """
    chars = set()
    for word in words:
        chars.update(word.lower())
    
    # Add padding and unknown tokens
    char_to_idx = {'<pad>': 0, '<unk>': 1}
    for i, char in enumerate(sorted(chars)):
        char_to_idx[char] = i + 2
        
    return char_to_idx
