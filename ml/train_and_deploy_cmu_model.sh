#!/bin/bash
# Train, evaluate, and deploy the CMU-based syllable counter model

# Set the Python executable
PYTHON=.venv/Scripts/python.exe

# Create the models directory if it doesn't exist
mkdir -p models

# Step 0: Download the Kaggle dataset if it doesn't exist
echo "Step 0: Downloading the Kaggle dataset if it doesn't exist..."
mkdir -p data
if [ ! -f data/syllable_dictionary.csv ]; then
  $PYTHON scripts/download_kaggle_dataset.py
fi

# Step 1: Create the CMU dataset
echo "Step 1: Creating the CMU dataset..."
$PYTHON scripts/optimized_cmu_counter.py --mode create --input_path data/syllable_dictionary.csv --output_path data/syllable_dictionary_cmu.csv

# Step 2: Train the CMU model
echo "Step 2: Training the CMU model..."
$PYTHON scripts/train_cmu.py --balance_method undersample --max_syllables 9 --batch_size 64 --epochs 30 --learning_rate 0.001 --use_existing

# Step 3: Convert the CMU model to ONNX format
echo "Step 3: Converting the CMU model to ONNX format..."
$PYTHON scripts/convert_cmu_to_onnx.py

# Step 4: Update the web application
echo "Step 4: Updating the web application..."
$PYTHON scripts/update_web_app_cmu.py

echo "Done! The CMU model has been trained, evaluated, and deployed."
echo "You can now test it interactively with: $PYTHON scripts/interactive_test_cmu.py"
