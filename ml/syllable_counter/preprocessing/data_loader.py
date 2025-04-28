"""
Data loading and preprocessing utilities for syllable counting.
"""
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_cmudict(file_path: str) -> Dict[str, int]:
    """
    Load the CMU Pronouncing Dictionary and extract syllable counts.
    
    Args:
        file_path: Path to the CMUdict file
        
    Returns:
        Dictionary mapping words to syllable counts
    """
    word_to_syllables = {}
    
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith(';;;'):
                    continue
                    
                parts = line.strip().split()
                if not parts:
                    continue
                    
                word = parts[0].lower()
                # Remove the stress numbers from the pronunciation
                pronunciation = [p.strip('0123456789') for p in parts[1:]]
                
                # Count vowel sounds as syllables
                syllable_count = sum(1 for p in pronunciation if any(c in 'AEIOU' for c in p))
                
                # Remove word number suffix if present (e.g., WORD(2))
                if '(' in word:
                    word = word.split('(')[0]
                    
                word_to_syllables[word] = syllable_count
    except Exception as e:
        print(f"Error loading CMUdict: {e}")
        
    return word_to_syllables


def create_dataset(word_to_syllables: Dict[str, int], 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create training and testing datasets from the syllable dictionary.
    
    Args:
        word_to_syllables: Dictionary mapping words to syllable counts
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame({
        'word': list(word_to_syllables.keys()),
        'syllables': list(word_to_syllables.values())
    })
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    return train_df, test_df


def prepare_features(words: List[str]) -> pd.DataFrame:
    """
    Extract features from words for syllable prediction.
    
    Args:
        words: List of words to extract features from
        
    Returns:
        DataFrame with extracted features
    """
    features = []
    vowels = 'aeiouy'
    consonants = 'bcdfghjklmnpqrstvwxz'
    
    for word in words:
        word = word.lower()
        
        # Basic features
        num_chars = len(word)
        num_vowels = sum(1 for c in word if c in vowels)
        num_consonants = sum(1 for c in word if c in consonants)
        
        # Vowel sequences
        vowel_sequences = 0
        in_vowel_seq = False
        for c in word:
            if c in vowels:
                if not in_vowel_seq:
                    vowel_sequences += 1
                    in_vowel_seq = True
            else:
                in_vowel_seq = False
        
        # Special patterns
        ends_with_e = int(word.endswith('e'))
        ends_with_le = int(word.endswith('le') and len(word) > 2 and word[-3] not in vowels)
        ends_with_ed = int(word.endswith('ed') and len(word) > 2)
        
        features.append({
            'word': word,
            'num_chars': num_chars,
            'num_vowels': num_vowels,
            'num_consonants': num_consonants,
            'vowel_sequences': vowel_sequences,
            'ends_with_e': ends_with_e,
            'ends_with_le': ends_with_le,
            'ends_with_ed': ends_with_ed,
        })
    
    return pd.DataFrame(features)
