"""
Validate the syllable dictionary dataset by comparing with a rule-based syllable counter.
"""
import pandas as pd
import re
from pathlib import Path
import random
from tqdm import tqdm


def rule_based_syllable_count(word):
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
        'eye': 1,
        'eyes': 1,
        'queue': 1,
        'queues': 1,
        'quay': 1,
        'quays': 1,
        'business': 2,
        'businesses': 3,
        'colonel': 2,
        'colonels': 2,
        'island': 2,
        'islands': 2,
        'recipe': 3,
        'recipes': 3,
        'wednesday': 3,
        'wednesdays': 3,
        'area': 3,
        'areas': 3,
        'idea': 3,
        'ideas': 3,
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


def validate_dataset(data_path, output_path=None, sample_size=1000, threshold=1):
    """
    Validate the syllable dictionary dataset by comparing with a rule-based syllable counter.
    
    Args:
        data_path: Path to the syllable dictionary CSV file
        output_path: Path to save the validated dataset
        sample_size: Number of words to sample for manual verification
        threshold: Maximum allowed difference between dataset and rule-based counts
        
    Returns:
        DataFrame with validation results
    """
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Clean the data
    df = df.dropna(subset=['word', 'syl'])
    df['word'] = df['word'].astype(str)
    
    print(f"Validating {len(df)} words...")
    
    # Add rule-based syllable count
    df['rule_based_syl'] = df['word'].progress_apply(rule_based_syllable_count)
    
    # Calculate difference
    df['diff'] = df['syl'] - df['rule_based_syl']
    df['abs_diff'] = df['diff'].abs()
    
    # Identify potentially incorrect entries
    potentially_incorrect = df[df['abs_diff'] > threshold]
    print(f"Found {len(potentially_incorrect)} potentially incorrect entries ({len(potentially_incorrect)/len(df):.2%} of total)")
    
    # Sample words for manual verification
    sample_df = potentially_incorrect.sample(min(sample_size, len(potentially_incorrect)), random_state=42)
    
    # Print sample for manual verification
    print("\nSample of potentially incorrect entries:")
    for _, row in sample_df.head(20).iterrows():
        print(f"Word: {row['word']}, Dataset: {row['syl']}, Rule-based: {row['rule_based_syl']}, Diff: {row['diff']}")
    
    # Save results if output path is provided
    if output_path:
        # Save full validation results
        df.to_csv(output_path, index=False)
        print(f"Validation results saved to {output_path}")
        
        # Save potentially incorrect entries
        incorrect_path = output_path.replace('.csv', '_incorrect.csv')
        potentially_incorrect.to_csv(incorrect_path, index=False)
        print(f"Potentially incorrect entries saved to {incorrect_path}")
        
        # Save sample for manual verification
        sample_path = output_path.replace('.csv', '_sample.csv')
        sample_df.to_csv(sample_path, index=False)
        print(f"Sample for manual verification saved to {sample_path}")
    
    return df


def create_curated_dataset(data_path, output_path, manual_corrections=None):
    """
    Create a curated dataset with manually verified syllable counts.
    
    Args:
        data_path: Path to the validated dataset
        output_path: Path to save the curated dataset
        manual_corrections: Dictionary of manual corrections (word -> correct syllable count)
        
    Returns:
        DataFrame with curated dataset
    """
    print(f"Loading validated dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Apply manual corrections if provided
    if manual_corrections:
        print(f"Applying {len(manual_corrections)} manual corrections...")
        for word, correct_syl in manual_corrections.items():
            df.loc[df['word'] == word, 'syl'] = correct_syl
    
    # Filter out words with large differences between dataset and rule-based counts
    # unless they have been manually corrected
    if manual_corrections:
        manually_corrected_words = set(manual_corrections.keys())
        df_filtered = df[(df['abs_diff'] <= 1) | (df['word'].isin(manually_corrected_words))]
    else:
        df_filtered = df[df['abs_diff'] <= 1]
    
    print(f"Curated dataset contains {len(df_filtered)} words ({len(df_filtered)/len(df):.2%} of original)")
    
    # Save curated dataset
    df_filtered.to_csv(output_path, index=False)
    print(f"Curated dataset saved to {output_path}")
    
    return df_filtered


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    # Add tqdm to pandas apply
    tqdm.pandas()
    
    parser = argparse.ArgumentParser(description="Validate syllable dictionary dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/syllable_dictionary.csv",
        help="Path to the syllable dictionary CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/syllable_dictionary_validated.csv",
        help="Path to save the validated dataset"
    )
    parser.add_argument(
        "--curated_path",
        type=str,
        default="data/syllable_dictionary_curated.csv",
        help="Path to save the curated dataset"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of words to sample for manual verification"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Maximum allowed difference between dataset and rule-based counts"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    validated_df = validate_dataset(
        args.data_path,
        args.output_path,
        args.sample_size,
        args.threshold
    )
    
    # Create curated dataset
    # You can add manual corrections here
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
    
    curated_df = create_curated_dataset(
        args.output_path,
        args.curated_path,
        manual_corrections
    )
