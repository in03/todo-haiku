"""
Train a decision tree classifier for syllable detection.
Treats syllabification as a binary classification task at each character position.
"""
import argparse
import os
import re
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text

# Import the CMU syllable counter
import sys
# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Try to import from optimized version first, fall back to original if not available
try:
    from optimized_cmu_counter import count_syllables_cmu, download_cmudict
    print("Using optimized CMU syllable counter")
except ImportError:
    from cmu_syllable_counter import count_syllables_cmu, download_cmudict
    print("Using original CMU syllable counter")

# Load the CMU dictionary
def load_cmu_dict():
    """
    Load the CMU dictionary.

    Returns:
        Dictionary mapping words to pronunciations
    """
    from nltk.corpus import cmudict
    return cmudict.dict()

# Get syllable boundaries from CMU dictionary
def get_syllable_boundaries(word, cmu_dict):
    """
    Get syllable boundaries for a word using the CMU dictionary.
    Returns a list of binary labels (1 = syllable boundary, 0 = no boundary).
    """
    import re
    word = word.lower().strip()
    word = re.sub(r'[^\w\s]', '', word)

    if word not in cmu_dict:
        return None

    # Get the pronunciation
    phones = cmu_dict[word][0]  # Use first pronunciation

    # Find vowel phonemes (those containing a digit)
    vowel_indices = [i for i, phone in enumerate(phones) if re.search(r'\d', phone)]

    # Count syllables
    num_syllables = len(vowel_indices)

    # If no syllables found, return None
    if num_syllables == 0:
        return None

    # Find vowels in the word
    vowels = 'aeiouy'
    vowel_positions = [i for i, char in enumerate(word) if char in vowels]

    # If no vowels found, return None
    if not vowel_positions:
        return None

    # Initialize boundaries
    char_boundaries = [0] * len(word)

    # If there's only one syllable, mark the first vowel as the boundary
    if num_syllables == 1:
        char_boundaries[vowel_positions[0]] = 1
        return char_boundaries

    # Try to map syllables to vowel groups in the word
    # This is a better approximation than the simple char_per_phone approach

    # First, identify vowel groups in the word
    vowel_groups = []
    current_group = []

    for i, char in enumerate(word):
        if char in vowels:
            current_group.append(i)
        elif current_group:
            vowel_groups.append(current_group)
            current_group = []

    # Add the last group if it exists
    if current_group:
        vowel_groups.append(current_group)

    # If we have the same number of vowel groups as syllables, it's easy
    if len(vowel_groups) == num_syllables:
        for group in vowel_groups:
            char_boundaries[group[0]] = 1
        return char_boundaries

    # Otherwise, we need to distribute syllables across vowel groups
    # This is a more complex mapping problem

    # If we have more vowel groups than syllables, merge some groups
    if len(vowel_groups) > num_syllables:
        # Find vowel pairs that are likely to be diphthongs
        diphthongs = ['ai', 'ay', 'ea', 'ee', 'ei', 'ey', 'ie', 'oa', 'oe', 'oi', 'oo', 'ou', 'ow', 'oy', 'ue', 'ui']

        # Check for diphthongs and mark them
        for i in range(len(word) - 1):
            if word[i:i+2] in diphthongs:
                # Find which vowel group contains this position
                for j, group in enumerate(vowel_groups):
                    if i in group and i+1 in group:
                        # This is a diphthong, mark only the first vowel
                        char_boundaries[i] = 1
                        break

        # If we still have more vowel groups than syllables, use a simple mapping
        if sum(char_boundaries) < num_syllables:
            # Reset boundaries
            char_boundaries = [0] * len(word)

            # Use a simple mapping: distribute syllables evenly across vowel groups
            step = len(vowel_groups) / num_syllables
            for i in range(num_syllables):
                group_idx = min(int(i * step), len(vowel_groups) - 1)
                group = vowel_groups[group_idx]
                char_boundaries[group[0]] = 1

    # If we have fewer vowel groups than syllables, split some groups
    elif len(vowel_groups) < num_syllables:
        # First, mark all vowel groups
        for group in vowel_groups:
            char_boundaries[group[0]] = 1

        # Then, find consonant clusters that might separate syllables
        consonant_clusters = []
        current_cluster = []

        for i, char in enumerate(word):
            if char not in vowels:
                current_cluster.append(i)
            elif current_cluster:
                if len(current_cluster) >= 2:  # Only consider clusters of 2+ consonants
                    consonant_clusters.append(current_cluster)
                current_cluster = []

        # Add the last cluster if it exists
        if current_cluster and len(current_cluster) >= 2:
            consonant_clusters.append(current_cluster)

        # Sort clusters by length (longest first)
        consonant_clusters.sort(key=len, reverse=True)

        # Add boundaries at consonant clusters until we have enough syllables
        for cluster in consonant_clusters:
            if sum(char_boundaries) >= num_syllables:
                break

            # Add a boundary after the first consonant in the cluster
            mid_idx = cluster[0] + 1
            if mid_idx < len(word) and char_boundaries[mid_idx] == 0:
                char_boundaries[mid_idx] = 1

    # Ensure we have exactly the right number of syllables
    while sum(char_boundaries) < num_syllables:
        # Find the longest gap between boundaries and add one in the middle
        max_gap = 0
        max_gap_start = 0

        current_gap = 0
        current_start = 0

        for i, boundary in enumerate(char_boundaries):
            if boundary == 1:
                if current_gap > max_gap:
                    max_gap = current_gap
                    max_gap_start = current_start
                current_gap = 0
                current_start = i + 1
            else:
                current_gap += 1

        # Check the last gap
        if current_gap > max_gap:
            max_gap = current_gap
            max_gap_start = current_start

        # Add a boundary in the middle of the largest gap
        if max_gap > 1:
            mid_idx = max_gap_start + max_gap // 2
            if mid_idx < len(word):
                char_boundaries[mid_idx] = 1
        else:
            # No more gaps to fill, break to avoid infinite loop
            break

    # If we have too many syllables, remove some boundaries
    while sum(char_boundaries) > num_syllables:
        # Find the shortest distance between boundaries and remove one
        min_dist = float('inf')
        min_idx = -1

        prev_idx = -1
        for i, boundary in enumerate(char_boundaries):
            if boundary == 1:
                if prev_idx != -1:
                    dist = i - prev_idx
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = i
                prev_idx = i

        # Remove the boundary that's closest to another one
        if min_idx != -1:
            char_boundaries[min_idx] = 0
        else:
            # No more boundaries to remove, break to avoid infinite loop
            break

    return char_boundaries

# Extract features for a character position
def extract_features(word, pos):
    """
    Extract features for a character position in a word.

    Args:
        word: The word
        pos: The character position

    Returns:
        Dictionary of features
    """
    vowels = 'aeiouy'
    consonants = 'bcdfghjklmnpqrstvwxz'
    word = word.lower()

    # Basic features
    features = {
        'is_vowel': 1 if pos < len(word) and word[pos] in vowels else 0,
        'is_consonant': 1 if pos < len(word) and word[pos] not in vowels else 0,
        'is_first_char': 1 if pos == 0 else 0,
        'is_last_char': 1 if pos == len(word) - 1 else 0,
        'is_second_last_char': 1 if pos == len(word) - 2 else 0,
        'word_length': len(word),
        'char_position': pos,
        'char_position_norm': pos / max(1, len(word) - 1),  # Normalized position
    }

    # Current character features
    if pos < len(word):
        current_char = word[pos]
        features.update({
            'is_a': 1 if current_char == 'a' else 0,
            'is_e': 1 if current_char == 'e' else 0,
            'is_i': 1 if current_char == 'i' else 0,
            'is_o': 1 if current_char == 'o' else 0,
            'is_u': 1 if current_char == 'u' else 0,
            'is_y': 1 if current_char == 'y' else 0,
        })

    # Previous character features
    if pos > 0:
        prev_char = word[pos - 1]
        features.update({
            'prev_is_vowel': 1 if prev_char in vowels else 0,
            'prev_is_consonant': 1 if prev_char not in vowels else 0,
            'prev_is_a': 1 if prev_char == 'a' else 0,
            'prev_is_e': 1 if prev_char == 'e' else 0,
            'prev_is_i': 1 if prev_char == 'i' else 0,
            'prev_is_o': 1 if prev_char == 'o' else 0,
            'prev_is_u': 1 if prev_char == 'u' else 0,
            'prev_is_y': 1 if prev_char == 'y' else 0,
        })
    else:
        features.update({
            'prev_is_vowel': 0,
            'prev_is_consonant': 0,
            'prev_is_a': 0,
            'prev_is_e': 0,
            'prev_is_i': 0,
            'prev_is_o': 0,
            'prev_is_u': 0,
            'prev_is_y': 0,
        })

    # Next character features
    if pos < len(word) - 1:
        next_char = word[pos + 1]
        features.update({
            'next_is_vowel': 1 if next_char in vowels else 0,
            'next_is_consonant': 1 if next_char not in vowels else 0,
            'next_is_a': 1 if next_char == 'a' else 0,
            'next_is_e': 1 if next_char == 'e' else 0,
            'next_is_i': 1 if next_char == 'i' else 0,
            'next_is_o': 1 if next_char == 'o' else 0,
            'next_is_u': 1 if next_char == 'u' else 0,
            'next_is_y': 1 if next_char == 'y' else 0,
        })
    else:
        features.update({
            'next_is_vowel': 0,
            'next_is_consonant': 0,
            'next_is_a': 0,
            'next_is_e': 0,
            'next_is_i': 0,
            'next_is_o': 0,
            'next_is_u': 0,
            'next_is_y': 0,
        })

    # Two characters before
    if pos > 1:
        prev2_char = word[pos - 2]
        features.update({
            'prev2_is_vowel': 1 if prev2_char in vowels else 0,
            'prev2_is_consonant': 1 if prev2_char not in vowels else 0,
        })
    else:
        features.update({
            'prev2_is_vowel': 0,
            'prev2_is_consonant': 0,
        })

    # Two characters after
    if pos < len(word) - 2:
        next2_char = word[pos + 2]
        features.update({
            'next2_is_vowel': 1 if next2_char in vowels else 0,
            'next2_is_consonant': 1 if next2_char not in vowels else 0,
        })
    else:
        features.update({
            'next2_is_vowel': 0,
            'next2_is_consonant': 0,
        })

    # Count vowels and consonants before and after current position
    vowels_before = sum(1 for i in range(pos) if i < len(word) and word[i] in vowels)
    consonants_before = pos - vowels_before
    vowels_after = sum(1 for i in range(pos + 1, len(word)) if word[i] in vowels)
    consonants_after = len(word) - pos - 1 - vowels_after

    features.update({
        'vowels_before': vowels_before,
        'consonants_before': consonants_before,
        'vowels_after': vowels_after,
        'consonants_after': consonants_after,
        'vowels_before_ratio': vowels_before / max(1, pos),
        'consonants_before_ratio': consonants_before / max(1, pos),
        'vowels_after_ratio': vowels_after / max(1, len(word) - pos - 1),
        'consonants_after_ratio': consonants_after / max(1, len(word) - pos - 1),
    })

    # Special patterns
    # Check for specific patterns
    if pos < len(word) - 1:
        features['is_le_ending'] = 1 if word[pos:pos+2] == 'le' and (pos == 0 or word[pos-1] not in vowels) else 0
        features['is_ed_ending'] = 1 if word[pos:pos+2] == 'ed' and pos == len(word) - 2 else 0
        features['is_es_ending'] = 1 if word[pos:pos+2] == 'es' and pos == len(word) - 2 else 0
        features['is_ing_ending'] = 1 if pos <= len(word) - 3 and word[pos:pos+3] == 'ing' and pos == len(word) - 3 else 0
    else:
        features['is_le_ending'] = 0
        features['is_ed_ending'] = 0
        features['is_es_ending'] = 0
        features['is_ing_ending'] = 0

    # Vowel sequence features
    if pos > 0 and pos < len(word):
        features['vowel_to_consonant'] = 1 if word[pos-1] in vowels and word[pos] not in vowels else 0
        features['consonant_to_vowel'] = 1 if word[pos-1] not in vowels and word[pos] in vowels else 0
    else:
        features['vowel_to_consonant'] = 0
        features['consonant_to_vowel'] = 0

    # Vowel pairs and triplets
    if pos < len(word) - 1:
        features['is_vowel_pair'] = 1 if pos < len(word) and pos + 1 < len(word) and word[pos] in vowels and word[pos+1] in vowels else 0
        if pos < len(word) - 2:
            features['is_vowel_triplet'] = 1 if word[pos] in vowels and word[pos+1] in vowels and word[pos+2] in vowels else 0
        else:
            features['is_vowel_triplet'] = 0
    else:
        features['is_vowel_pair'] = 0
        features['is_vowel_triplet'] = 0

    # Consonant pairs and triplets
    if pos < len(word) - 1:
        features['is_consonant_pair'] = 1 if pos < len(word) and pos + 1 < len(word) and word[pos] in consonants and word[pos+1] in consonants else 0
        if pos < len(word) - 2:
            features['is_consonant_triplet'] = 1 if word[pos] in consonants and word[pos+1] in consonants and word[pos+2] in consonants else 0
        else:
            features['is_consonant_triplet'] = 0
    else:
        features['is_consonant_pair'] = 0
        features['is_consonant_triplet'] = 0

    # Common syllable patterns
    common_prefixes = ['re', 'un', 'in', 'im', 'dis', 'pre', 'pro', 'con']
    common_suffixes = ['ly', 'er', 'ing', 'ed', 'es', 'ion', 'tion', 'ment', 'ness', 'ful', 'less']

    # Check for prefixes
    for prefix in common_prefixes:
        prefix_len = len(prefix)
        if pos == prefix_len - 1 and word[:prefix_len] == prefix:
            features[f'after_{prefix}_prefix'] = 1
            break
    else:
        features['after_common_prefix'] = 0

    # Check for suffixes
    for suffix in common_suffixes:
        suffix_len = len(suffix)
        if len(word) >= suffix_len and pos == len(word) - suffix_len - 1 and word[-suffix_len:] == suffix:
            features[f'before_{suffix}_suffix'] = 1
            break
    else:
        features['before_common_suffix'] = 0

    return features

# Prepare dataset for decision tree training
def prepare_dataset(words, cmu_dict):
    """
    Prepare dataset for decision tree training.

    Args:
        words: List of words
        cmu_dict: CMU dictionary

    Returns:
        X, y, feature_names: Features, labels, and feature names
    """
    X = []
    y = []
    feature_names = None

    for word in tqdm(words, desc="Preparing dataset"):
        boundaries = get_syllable_boundaries(word, cmu_dict)
        if boundaries is None:
            continue

        for pos in range(len(word)):
            features = extract_features(word, pos)

            # Store feature names from the first sample
            if feature_names is None:
                feature_names = list(features.keys())

            X.append([features[name] for name in feature_names])
            y.append(boundaries[pos])

    return np.array(X), np.array(y), feature_names

# Convert decision tree to JSON
def tree_to_json(tree, feature_names):
    """
    Convert a scikit-learn decision tree to a JSON representation.

    Args:
        tree: Trained decision tree
        feature_names: List of feature names

    Returns:
        JSON representation of the tree
    """
    js_tree = {}

    def recurse(node, tree_json):
        if tree.children_left[node] == -1:  # Leaf node
            tree_json['value'] = int(np.argmax(tree.value[node][0]))
            tree_json['isLeaf'] = True
            return

        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]

        tree_json['feature'] = feature
        tree_json['threshold'] = threshold
        tree_json['isLeaf'] = False

        # Left child (feature <= threshold)
        tree_json['left'] = {}
        recurse(tree.children_left[node], tree_json['left'])

        # Right child (feature > threshold)
        tree_json['right'] = {}
        recurse(tree.children_right[node], tree_json['right'])

    recurse(0, js_tree)
    return js_tree

# Train decision tree model
def train_model(
    data_path,
    model_dir,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_samples=None,
    use_grid_search=True,
    n_jobs=-1
):
    """
    Train a decision tree classifier for syllable detection.

    Args:
        data_path: Path to the syllable dictionary CSV file
        model_dir: Directory to save the model
        max_depth: Maximum depth of the decision tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        max_samples: Maximum number of samples to use (for debugging)
        use_grid_search: Whether to use grid search for hyperparameter tuning
        n_jobs: Number of jobs to run in parallel for grid search
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
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

    # Download CMU dict
    download_cmudict()

    # Load CMU dict
    cmu_dict = load_cmu_dict()

    # Prepare dataset
    print("Preparing dataset...")
    X, y, feature_names = prepare_dataset(words, cmu_dict)

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"Class {value}: {count} samples ({count/len(y):.2%})")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train decision tree
    if use_grid_search:
        print("Performing grid search for hyperparameter tuning...")
        from sklearn.model_selection import GridSearchCV

        # Define parameter grid
        param_grid = {
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced']
        }

        # Create grid search
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )

        # Fit grid search
        grid_search.fit(X_train, y_train)

        # Get best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # Train model with best parameters
        clf = DecisionTreeClassifier(random_state=42, **best_params)
        clf.fit(X_train, y_train)
    else:
        print("Training decision tree classifier...")
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Print feature importances
    print("\nFeature Importances (top 20):")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(20, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # Print tree structure (first few levels)
    tree_text = export_text(clf, feature_names=feature_names, max_depth=5)
    print("\nTree Structure (first 5 levels):")
    print(tree_text)

    # Convert tree to JSON
    js_tree = tree_to_json(clf.tree_, feature_names)

    # Save model
    with open(os.path.join(model_dir, 'syllable_decision_tree.json'), 'w') as f:
        json.dump(js_tree, f, indent=2)

    # Save feature names
    with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)

    # Save model metadata
    metadata = {
        'accuracy': accuracy,
        'feature_names': feature_names
    }

    # Add model parameters
    if use_grid_search:
        metadata.update(best_params)
    else:
        metadata.update({
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        })

    with open(os.path.join(model_dir, 'decision_tree_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Test on some common words
    test_words = [
        'hello', 'world', 'python', 'syllable', 'counter', 'decision', 'tree',
        'haiku', 'poetry', 'japanese', 'tradition', 'seventeen', 'syllables',
        'five', 'seven', 'nature', 'season', 'moment', 'insight',
        'supercalifragilisticexpialidocious'
    ]

    print("\nTesting on common words:")
    for word in test_words:
        # Get actual syllable count
        actual_count = count_syllables_cmu(word)
        if actual_count is None:
            actual_count = "Unknown"

        # Predict syllable boundaries
        boundaries = []
        for pos in range(len(word)):
            features = extract_features(word, pos)
            feature_values = [features[name] for name in feature_names]
            prediction = clf.predict([feature_values])[0]
            boundaries.append(prediction)

        # Ensure the first syllable starts at the beginning
        if 1 in boundaries:
            boundaries[0] = 1

        # Count syllables
        predicted_count = sum(boundaries)

        print(f"Word: {word}")
        print(f"Predicted boundaries: {boundaries}")
        print(f"Predicted count: {predicted_count}")
        print(f"Actual count: {actual_count}")
        print()

    print(f"Model saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train decision tree classifier for syllable detection")
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
        "--max_depth",
        type=int,
        default=15,
        help="Maximum depth of the decision tree"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum samples required to split a node"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=1,
        help="Minimum samples required at a leaf node"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for debugging)"
    )
    parser.add_argument(
        "--use_grid_search",
        action="store_true",
        help="Whether to use grid search for hyperparameter tuning"
    )
    parser.add_argument(
        "--no_grid_search",
        action="store_false",
        dest="use_grid_search",
        help="Disable grid search for hyperparameter tuning"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs to run in parallel for grid search (-1 for all cores)"
    )
    parser.set_defaults(use_grid_search=True)

    args = parser.parse_args()
    train_model(**vars(args))
