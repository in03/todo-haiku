"""
Convert a trained decision tree model to JavaScript code.
"""
import argparse
import os
import json
from pathlib import Path

def generate_js_code(tree_json, feature_names):
    """
    Generate JavaScript code from a decision tree JSON.
    
    Args:
        tree_json: JSON representation of the decision tree
        feature_names: List of feature names
        
    Returns:
        JavaScript code as a string
    """
    # Start with the feature extraction function
    js_code = """// Decision tree-based syllable counter
// Auto-generated from a trained scikit-learn decision tree

// Extract features for a character position
function extractFeatures(word, pos) {
  const vowels = 'aeiouy';
  word = word.toLowerCase();
  
  // Basic features
  const features = {
    is_vowel: pos < word.length && vowels.includes(word[pos]) ? 1 : 0,
    is_consonant: pos < word.length && !vowels.includes(word[pos]) ? 1 : 0,
    is_first_char: pos === 0 ? 1 : 0,
    is_last_char: pos === word.length - 1 ? 1 : 0,
    word_length: word.length,
    char_position: pos,
    char_position_norm: word.length > 1 ? pos / (word.length - 1) : 0,
  };
  
  // Previous character features
  if (pos > 0) {
    const prevChar = word[pos - 1];
    features.prev_is_vowel = vowels.includes(prevChar) ? 1 : 0;
    features.prev_is_consonant = !vowels.includes(prevChar) ? 1 : 0;
  } else {
    features.prev_is_vowel = 0;
    features.prev_is_consonant = 0;
  }
  
  // Next character features
  if (pos < word.length - 1) {
    const nextChar = word[pos + 1];
    features.next_is_vowel = vowels.includes(nextChar) ? 1 : 0;
    features.next_is_consonant = !vowels.includes(nextChar) ? 1 : 0;
  } else {
    features.next_is_vowel = 0;
    features.next_is_consonant = 0;
  }
  
  // Two characters before
  if (pos > 1) {
    const prev2Char = word[pos - 2];
    features.prev2_is_vowel = vowels.includes(prev2Char) ? 1 : 0;
    features.prev2_is_consonant = !vowels.includes(prev2Char) ? 1 : 0;
  } else {
    features.prev2_is_vowel = 0;
    features.prev2_is_consonant = 0;
  }
  
  // Two characters after
  if (pos < word.length - 2) {
    const next2Char = word[pos + 2];
    features.next2_is_vowel = vowels.includes(next2Char) ? 1 : 0;
    features.next2_is_consonant = !vowels.includes(next2Char) ? 1 : 0;
  } else {
    features.next2_is_vowel = 0;
    features.next2_is_consonant = 0;
  }
  
  // Special patterns
  if (pos < word.length) {
    const currentChar = word[pos];
    features.is_e = currentChar === 'e' ? 1 : 0;
    features.is_y = currentChar === 'y' ? 1 : 0;
  }
  
  // Check for specific patterns
  if (pos < word.length - 1) {
    features.is_le_ending = word.substring(pos, pos + 2) === 'le' && 
                           (pos === 0 || !vowels.includes(word[pos - 1])) ? 1 : 0;
    features.is_ed_ending = word.substring(pos, pos + 2) === 'ed' && 
                           pos === word.length - 2 ? 1 : 0;
  } else {
    features.is_le_ending = 0;
    features.is_ed_ending = 0;
  }
  
  // Vowel sequence features
  if (pos > 0 && pos < word.length) {
    features.vowel_to_consonant = vowels.includes(word[pos - 1]) && 
                                 !vowels.includes(word[pos]) ? 1 : 0;
    features.consonant_to_vowel = !vowels.includes(word[pos - 1]) && 
                                 vowels.includes(word[pos]) ? 1 : 0;
  } else {
    features.vowel_to_consonant = 0;
    features.consonant_to_vowel = 0;
  }
  
  return features;
}

// Decision tree model as JSON
const decisionTree = """
    
    # Add the tree JSON
    js_code += json.dumps(tree_json, indent=2)
    
    # Add the prediction function
    js_code += """;

// Predict syllable boundary using the decision tree
function predictBoundary(features) {
  let node = decisionTree;
  
  while (!node.isLeaf) {
    const featureValue = features[node.feature];
    
    if (featureValue <= node.threshold) {
      node = node.left;
    } else {
      node = node.right;
    }
  }
  
  return node.value;
}

// Get syllable boundaries for a word
function getSyllableBoundaries(word) {
  if (!word) return [];
  
  const boundaries = [];
  
  for (let pos = 0; pos < word.length; pos++) {
    const features = extractFeatures(word, pos);
    const isBoundary = predictBoundary(features);
    boundaries.push(isBoundary);
  }
  
  // Ensure the first syllable starts at the beginning
  if (boundaries.indexOf(1) !== -1) {
    boundaries[0] = 1;
  }
  
  return boundaries;
}

// Count syllables in a word
function countSyllablesInWord(word) {
  if (!word) return 0;
  
  // Remove punctuation and convert to lowercase
  word = word.toLowerCase().replace(/[.,;:!?()'"]/g, '');
  
  // Special cases dictionary
  const specialCases = {
    'eye': 1, 'eyes': 1, 'queue': 1, 'queues': 1, 'quay': 1, 'quays': 1,
    'business': 2, 'businesses': 3, 'colonel': 2, 'colonels': 2,
    'island': 2, 'islands': 2, 'recipe': 3, 'recipes': 3,
    'wednesday': 3, 'wednesdays': 3, 'area': 3, 'areas': 3,
    'idea': 3, 'ideas': 3,
  };
  
  // Check for special cases
  if (specialCases[word] !== undefined) {
    return specialCases[word];
  }
  
  // Get syllable boundaries
  const boundaries = getSyllableBoundaries(word);
  
  // Count boundaries (each boundary marks the start of a syllable)
  const count = boundaries.reduce((sum, val) => sum + val, 0);
  
  // Ensure at least one syllable
  return Math.max(1, count);
}

// Count syllables in a line of text
function countSyllables(text) {
  if (!text) return 0;
  
  // Split text into words
  const words = text.split(/\\s+/).filter(word => word.length > 0);
  
  // Count syllables in each word and sum them
  return words.reduce((total, word) => total + countSyllablesInWord(word), 0);
}

// Export functions
export {
  countSyllables,
  countSyllablesInWord,
  getSyllableBoundaries
};
"""
    
    return js_code

def convert_to_js(model_dir, output_path):
    """
    Convert a trained decision tree model to JavaScript code.
    
    Args:
        model_dir: Directory containing the model files
        output_path: Path to save the JavaScript code
    """
    # Load the tree JSON
    tree_path = os.path.join(model_dir, 'syllable_decision_tree.json')
    with open(tree_path, 'r') as f:
        tree_json = json.load(f)
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    # Generate JavaScript code
    js_code = generate_js_code(tree_json, feature_names)
    
    # Save JavaScript code
    with open(output_path, 'w') as f:
        f.write(js_code)
    
    print(f"JavaScript code saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert decision tree model to JavaScript")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing the model files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../src/utils/decision-tree-syllable-counter.ts",
        help="Path to save the JavaScript code"
    )
    
    args = parser.parse_args()
    convert_to_js(args.model_dir, args.output_path)
