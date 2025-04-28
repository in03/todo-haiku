"""
Data loading and preprocessing utilities for the Kaggle syllable dictionary.
"""
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def load_kaggle_dataset(file_path: str) -> Dict[str, int]:
    """
    Load the Kaggle English Phonetic and Syllable Count Dictionary.

    Args:
        file_path: Path to the Kaggle dataset CSV file

    Returns:
        Dictionary mapping words to syllable counts
    """
    word_to_syllables = {}

    try:
        df = pd.read_csv(file_path)
        print(f"CSV columns: {df.columns.tolist()}")

        # Use 'syl' column instead of 'syllable_count'
        for _, row in df.iterrows():
            word = str(row['word']).lower()
            # Ensure the word is valid
            if not word or not isinstance(word, str):
                continue

            # Get syllable count from 'syl' column
            syllable_count = row['syl']
            if not pd.isna(syllable_count) and syllable_count > 0:
                word_to_syllables[word] = int(syllable_count)
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")

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

    # Print statistics before filtering
    total_words = len(df)
    print(f"Total words before filtering: {total_words}")
    print(f"Syllable count distribution: {df['syllables'].value_counts().sort_index()}")

    # Filter out words with too many syllables (rare cases)
    df = df[df['syllables'] < 10]

    # Print statistics after filtering
    filtered_words = len(df)
    print(f"Words after filtering: {filtered_words} ({filtered_words/total_words:.2%} of total)")

    # Split into train and test sets
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

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
