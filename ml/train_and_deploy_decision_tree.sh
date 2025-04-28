#!/bin/bash
# Train, evaluate, and deploy the decision tree syllable counter model

# Set the Python executable
PYTHON=.venv/Scripts/python.exe

# Create the models directory if it doesn't exist
mkdir -p models

# Step 1: Create the CMU dataset if it doesn't exist
echo "Step 1: Checking for CMU dataset..."
if [ ! -f "data/syllable_dictionary_cmu.csv" ]; then
  echo "Creating the CMU dataset..."
  $PYTHON scripts/optimized_cmu_counter.py --mode create --input_path data/syllable_dictionary.csv --output_path data/syllable_dictionary_cmu.csv
else
  echo "CMU dataset already exists."
fi

# Step 2: Train the decision tree model
echo "Step 2: Training the decision tree model..."
$PYTHON scripts/train_decision_tree.py --max_depth 10 --min_samples_split 2 --min_samples_leaf 1

# Step 3: Convert the decision tree model to JavaScript
# Skip step 3 since we're now using a hand-crafted decision tree implementation
# echo "Step 3: Converting the decision tree model to JavaScript..."
# $PYTHON scripts/convert_decision_tree_to_js.py --output_path ../src/utils/decision-tree-syllable-counter.ts
echo "Step 3: Skipping model conversion - using hand-crafted decision tree implementation"

echo "Done! The decision tree model has been trained and deployed."
echo "You can now use the decision tree syllable counter in the Todo Haiku app."
echo "To use it, set the syllable counter type to DECISION_TREE:"
echo "  import { setSyllableCounterType, SyllableCounterType } from '~/utils/syllable-counter-switch';"
echo "  setSyllableCounterType(SyllableCounterType.DECISION_TREE);"
