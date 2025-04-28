"""
Test script for the decision tree syllable counter.
This script trains a small decision tree model and tests it on a few examples.
"""
import os
import sys
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import from optimized version first, fall back to original if not available
try:
    from optimized_cmu_counter import count_syllables_cmu, download_cmudict, load_cmu_dict
    print("Using optimized CMU syllable counter")
except ImportError:
    from cmu_syllable_counter import count_syllables_cmu, download_cmudict
    print("Using original CMU syllable counter")
    # Define load_cmu_dict if not available
    def load_cmu_dict():
        from nltk.corpus import cmudict
        return cmudict.dict()

# Vowels for feature extraction
VOWELS = 'aeiouy'

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
    word = word.lower()
    
    # Basic features
    features = {
        'is_vowel': 1 if pos < len(word) and word[pos] in VOWELS else 0,
        'is_consonant': 1 if pos < len(word) and word[pos] not in VOWELS else 0,
        'is_first_char': 1 if pos == 0 else 0,
        'is_last_char': 1 if pos == len(word) - 1 else 0,
        'word_length': len(word),
        'char_position': pos,
        'char_position_norm': pos / max(1, len(word) - 1),  # Normalized position
    }
    
    # Previous character features
    if pos > 0:
        prev_char = word[pos - 1]
        features.update({
            'prev_is_vowel': 1 if prev_char in VOWELS else 0,
            'prev_is_consonant': 1 if prev_char not in VOWELS else 0,
        })
    else:
        features.update({
            'prev_is_vowel': 0,
            'prev_is_consonant': 0,
        })
    
    # Next character features
    if pos < len(word) - 1:
        next_char = word[pos + 1]
        features.update({
            'next_is_vowel': 1 if next_char in VOWELS else 0,
            'next_is_consonant': 1 if next_char not in VOWELS else 0,
        })
    else:
        features.update({
            'next_is_vowel': 0,
            'next_is_consonant': 0,
        })
    
    # Vowel sequence features
    if pos > 0 and pos < len(word):
        features['vowel_to_consonant'] = 1 if word[pos-1] in VOWELS and word[pos] not in VOWELS else 0
        features['consonant_to_vowel'] = 1 if word[pos-1] not in VOWELS and word[pos] in VOWELS else 0
    else:
        features['vowel_to_consonant'] = 0
        features['consonant_to_vowel'] = 0
    
    return features

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
    
    # Map phoneme indices to character indices
    # This is an approximation since there's not a 1:1 mapping
    char_boundaries = [0] * (len(word) + 1)  # +1 because we need boundaries between chars
    
    # Simple mapping strategy: distribute vowels evenly across the word
    char_per_phone = len(word) / len(phones)
    for i in vowel_indices:
        char_idx = min(int(i * char_per_phone), len(word) - 1)
        char_boundaries[char_idx] = 1
    
    # First vowel always starts a syllable
    if 1 in char_boundaries:
        first_vowel = char_boundaries.index(1)
        char_boundaries[first_vowel] = 1
    
    return char_boundaries[:-1]  # Remove the last boundary (after the last char)

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
    
    for word in words:
        boundaries = get_syllable_boundaries(word, cmu_dict)
        if boundaries is None:
            continue
        
        for pos in range(len(word)):
            features = extract_features(word, pos)
            X.append(list(features.values()))
            y.append(boundaries[pos])
    
    feature_names = list(extract_features(words[0], 0).keys())
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

# Predict syllable boundaries using the decision tree
def predict_boundaries(word, clf, feature_names):
    """
    Predict syllable boundaries for a word using the decision tree.
    
    Args:
        word: The word
        clf: Trained decision tree classifier
        feature_names: List of feature names
        
    Returns:
        List of binary labels (1 = syllable boundary, 0 = no boundary)
    """
    boundaries = []
    
    for pos in range(len(word)):
        features = extract_features(word, pos)
        feature_values = [features[name] for name in feature_names]
        prediction = clf.predict([feature_values])[0]
        boundaries.append(prediction)
    
    # Ensure the first syllable starts at the beginning
    if 1 in boundaries:
        boundaries[0] = 1
    
    return boundaries

# Count syllables from boundaries
def count_syllables_from_boundaries(boundaries):
    """
    Count syllables from boundaries.
    
    Args:
        boundaries: List of binary labels (1 = syllable boundary, 0 = no boundary)
        
    Returns:
        Number of syllables
    """
    return sum(boundaries)

# Test the decision tree on a word
def test_word(word, clf, feature_names, cmu_dict):
    """
    Test the decision tree on a word.
    
    Args:
        word: The word to test
        clf: Trained decision tree classifier
        feature_names: List of feature names
        cmu_dict: CMU dictionary
        
    Returns:
        Predicted syllable count, actual syllable count
    """
    # Get actual syllable count
    actual_count = count_syllables_cmu(word)
    if actual_count is None:
        actual_count = "Unknown"
    
    # Predict syllable boundaries
    boundaries = predict_boundaries(word, clf, feature_names)
    
    # Count syllables
    predicted_count = count_syllables_from_boundaries(boundaries)
    
    return predicted_count, actual_count, boundaries

def main():
    # Download CMU dict
    download_cmudict()
    
    # Load CMU dict
    cmu_dict = load_cmu_dict()
    
    # Sample words for training
    sample_words = [
        'hello', 'world', 'python', 'syllable', 'counter', 'decision', 'tree',
        'classifier', 'machine', 'learning', 'artificial', 'intelligence',
        'computer', 'science', 'programming', 'language', 'algorithm',
        'data', 'structure', 'function', 'method', 'class', 'object',
        'variable', 'constant', 'expression', 'statement', 'condition',
        'loop', 'iteration', 'recursion', 'parameter', 'argument',
        'return', 'value', 'type', 'string', 'integer', 'float', 'boolean',
        'array', 'list', 'dictionary', 'set', 'tuple', 'module', 'package',
        'library', 'framework', 'api', 'interface', 'implementation',
        'abstract', 'concrete', 'inheritance', 'polymorphism', 'encapsulation',
        'abstraction', 'composition', 'aggregation', 'association', 'dependency'
    ]
    
    # Prepare dataset
    X, y, feature_names = prepare_dataset(sample_words, cmu_dict)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train decision tree
    print("Training decision tree classifier...")
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # Print tree structure
    tree_text = export_text(clf, feature_names=feature_names)
    print("\nTree Structure:")
    print(tree_text)
    
    # Convert tree to JSON
    js_tree = tree_to_json(clf.tree_, feature_names)
    print("\nJSON Tree:")
    print(json.dumps(js_tree, indent=2))
    
    # Test on some words
    test_words = [
        'hello', 'world', 'python', 'syllable', 'counter', 'decision', 'tree',
        'haiku', 'poetry', 'japanese', 'tradition', 'seventeen', 'syllables',
        'five', 'seven', 'five', 'nature', 'season', 'moment', 'insight',
        'supercalifragilisticexpialidocious'
    ]
    
    print("\nTesting on words:")
    for word in test_words:
        predicted_count, actual_count, boundaries = test_word(word, clf, feature_names, cmu_dict)
        print(f"Word: {word}")
        print(f"Predicted boundaries: {boundaries}")
        print(f"Predicted count: {predicted_count}")
        print(f"Actual count: {actual_count}")
        print()

if __name__ == "__main__":
    main()
