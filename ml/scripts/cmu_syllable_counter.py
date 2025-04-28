"""
Syllable counter based on the CMU Pronouncing Dictionary.
"""
import re
import os
import nltk
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

# Download the CMU Pronouncing Dictionary if not already downloaded
def download_cmudict():
    try:
        nltk.data.find('corpora/cmudict')
    except LookupError:
        print("Downloading CMU Pronouncing Dictionary...")
        nltk.download('cmudict')

# Count syllables using the CMU Pronouncing Dictionary
def count_syllables_cmu(word):
    """
    Count syllables using the CMU Pronouncing Dictionary.

    Args:
        word: Word to count syllables for

    Returns:
        Number of syllables or None if word not in dictionary
    """
    word = word.lower().strip()

    # Remove punctuation
    word = re.sub(r'[^\w\s]', '', word)

    # Get pronunciations from CMU dict
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()

        if word in d:
            # Count vowel phonemes (those containing a digit)
            return max([len([ph for ph in phones if re.search(r'\d', ph)]) for phones in d[word]])
        else:
            return None
    except Exception as e:
        print(f"Error using CMU dict: {e}")
        return None

# Rule-based syllable counter as fallback
def count_syllables_rule_based(word):
    """
    Count syllables using a rule-based approach.

    Args:
        word: Word to count syllables for

    Returns:
        Number of syllables
    """
    # Remove punctuation and convert to lowercase
    word = re.sub(r'[.,;:!?()\'"]', '', word.lower())

    # Special cases dictionary
    special_cases = {
        'eye': 1, 'eyes': 1, 'queue': 1, 'queues': 1, 'quay': 1, 'quays': 1,
        'business': 2, 'businesses': 3, 'colonel': 2, 'colonels': 2,
        'island': 2, 'islands': 2, 'recipe': 3, 'recipes': 3,
        'wednesday': 3, 'wednesdays': 3, 'area': 3, 'areas': 3,
        'idea': 3, 'ideas': 3,
    }

    # Check for special cases
    if word in special_cases:
        return special_cases[word]

    # Count syllables based on vowel groups
    vowels = 'aeiouy'
    count = 0
    prev_is_vowel = False

    # Handle specific patterns
    if word and vowels.find(word[0]) >= 0:
        count = 1
        prev_is_vowel = True

    for i in range(1, len(word)):
        is_vowel = vowels.find(word[i]) >= 0

        if is_vowel and not prev_is_vowel:
            count += 1

        prev_is_vowel = is_vowel

    # Handle silent e at the end
    if len(word) > 2 and word.endswith('e') and not word[-2] in vowels:
        count = max(1, count - 1)

    # Handle words ending with 'le' where the 'l' is preceded by a consonant
    if len(word) > 2 and word.endswith('le') and not word[-3] in vowels:
        count += 1

    # Handle words ending with 'ed'
    if len(word) > 2 and word.endswith('ed'):
        # Only count as a syllable if preceded by t or d
        if word[-3] not in ['t', 'd']:
            count = max(1, count - 1)

    # Ensure at least one syllable
    return max(1, count)

# Hybrid syllable counter that uses CMU dict when available, falls back to rule-based
def count_syllables_hybrid(word):
    """
    Count syllables using CMU dict when available, falling back to rule-based.

    Args:
        word: Word to count syllables for

    Returns:
        Number of syllables
    """
    cmu_count = count_syllables_cmu(word)
    if cmu_count is not None:
        return cmu_count
    else:
        return count_syllables_rule_based(word)

# Process the Kaggle dataset and create a corrected version
def process_kaggle_dataset(input_path, output_path):
    """
    Process the Kaggle dataset and create a corrected version using CMU dict.

    Args:
        input_path: Path to the Kaggle dataset
        output_path: Path to save the corrected dataset
    """
    # Initialize tqdm.pandas() to enable progress_apply
    from tqdm import tqdm
    tqdm.pandas()

    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Clean the data
    df = df.dropna(subset=['word', 'syl'])
    df['word'] = df['word'].astype(str)

    # Add CMU syllable count
    print("Counting syllables using CMU dict (this may take a while)...")
    df['cmu_syl'] = df['word'].progress_apply(count_syllables_cmu)

    # Add rule-based syllable count for words not in CMU dict
    print("Counting syllables using rule-based approach for words not in CMU dict...")
    df['hybrid_syl'] = df.apply(
        lambda row: row['cmu_syl'] if row['cmu_syl'] is not None else count_syllables_rule_based(row['word']),
        axis=1
    )

    # Calculate differences
    df['diff_kaggle_cmu'] = df.apply(
        lambda row: row['syl'] - row['cmu_syl'] if row['cmu_syl'] is not None else None,
        axis=1
    )
    df['diff_kaggle_hybrid'] = df['syl'] - df['hybrid_syl']

    # Print statistics
    print("\nStatistics:")
    print(f"Total words: {len(df)}")
    print(f"Words in CMU dict: {df['cmu_syl'].notna().sum()} ({df['cmu_syl'].notna().sum()/len(df):.2%})")
    print(f"Words with different Kaggle vs. CMU counts: {(df['diff_kaggle_cmu'] != 0).sum()} ({(df['diff_kaggle_cmu'] != 0).sum()/df['cmu_syl'].notna().sum():.2%})")
    print(f"Words with different Kaggle vs. Hybrid counts: {(df['diff_kaggle_hybrid'] != 0).sum()} ({(df['diff_kaggle_hybrid'] != 0).sum()/len(df):.2%})")

    # Print examples of differences
    print("\nExamples of differences between Kaggle and CMU counts:")
    diff_df = df[(df['diff_kaggle_cmu'] != 0) & df['cmu_syl'].notna()].sample(min(10, len(df[(df['diff_kaggle_cmu'] != 0) & df['cmu_syl'].notna()])))
    for _, row in diff_df.iterrows():
        print(f"Word: {row['word']}, Kaggle: {row['syl']}, CMU: {row['cmu_syl']}, Diff: {row['diff_kaggle_cmu']}")

    # Save corrected dataset
    print(f"\nSaving corrected dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

    return df

# Create a curated dataset using CMU dict
def create_cmu_dataset(input_path, output_path):
    """
    Create a curated dataset using CMU dict.

    Args:
        input_path: Path to the Kaggle dataset
        output_path: Path to save the curated dataset
    """
    # Initialize tqdm.pandas() to enable progress_apply
    from tqdm import tqdm
    tqdm.pandas()

    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # Clean the data
    df = df.dropna(subset=['word'])
    df['word'] = df['word'].astype(str)

    # Count syllables using hybrid approach
    print("Counting syllables using hybrid approach...")
    df['syllables'] = df['word'].progress_apply(count_syllables_hybrid)

    # Keep only necessary columns
    df_curated = df[['word', 'syllables']].copy()

    # Save curated dataset
    print(f"\nSaving curated dataset to {output_path}...")
    df_curated.to_csv(output_path, index=False)
    print("Done!")

    return df_curated

# Create a test dataset with words from the CMU dict
def create_test_dataset(output_path, num_words=1000):
    """
    Create a test dataset with words from the CMU dict.

    Args:
        output_path: Path to save the test dataset
        num_words: Number of words to include
    """
    from nltk.corpus import cmudict
    d = cmudict.dict()

    # Get a random sample of words
    import random
    words = random.sample(list(d.keys()), min(num_words, len(d)))

    # Count syllables
    syllables = [max([len([ph for ph in phones if re.search(r'\d', ph)]) for phones in d[word]]) for word in words]

    # Create DataFrame
    df = pd.DataFrame({'word': words, 'syllables': syllables})

    # Save test dataset
    print(f"\nSaving test dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

    return df

# Interactive testing
def interactive_test():
    """
    Interactive testing of syllable counters.
    """
    print("Interactive Syllable Counter")
    print("===========================")
    print("Enter words to count syllables (or 'q' to quit)")

    while True:
        word = input("\n> ").strip()

        if word.lower() == 'q':
            break

        cmu_count = count_syllables_cmu(word)
        rule_count = count_syllables_rule_based(word)
        hybrid_count = count_syllables_hybrid(word)

        print(f"CMU Dict: {cmu_count if cmu_count is not None else 'Not found'}")
        print(f"Rule-based: {rule_count}")
        print(f"Hybrid: {hybrid_count}")

if __name__ == "__main__":
    # Add tqdm to pandas apply
    tqdm.pandas()

    # Download CMU dict
    download_cmudict()

    parser = argparse.ArgumentParser(description="Syllable counter based on the CMU Pronouncing Dictionary")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['process', 'create', 'test', 'interactive'],
        default='interactive',
        help="Mode to run in"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/syllable_dictionary.csv",
        help="Path to the input dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/syllable_dictionary_cmu.csv",
        help="Path to save the output dataset"
    )
    parser.add_argument(
        "--num_words",
        type=int,
        default=1000,
        help="Number of words to include in test dataset"
    )

    args = parser.parse_args()

    if args.mode == 'process':
        process_kaggle_dataset(args.input_path, args.output_path)
    elif args.mode == 'create':
        create_cmu_dataset(args.input_path, args.output_path)
    elif args.mode == 'test':
        create_test_dataset(args.output_path, args.num_words)
    else:  # interactive
        interactive_test()
