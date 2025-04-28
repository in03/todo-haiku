#!/bin/bash
# Train, evaluate, and deploy the character-level LSTM syllable counter model

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

# Step 2: Train the character-level LSTM model
echo "Step 2: Training the character-level LSTM model..."
$PYTHON scripts/train_char_lstm.py --batch_size 64 --epochs 30 --learning_rate 0.001 --embedding_dim 64 --hidden_dim 128 --n_layers 2 --dropout 0.2 --early_stopping_patience 5

# Step 3: Convert the character-level LSTM model to ONNX format
echo "Step 3: Converting the character-level LSTM model to ONNX format..."
$PYTHON scripts/convert_char_lstm_to_onnx.py

# Step 4: Update the web application
echo "Step 4: Updating the web application..."
$PYTHON scripts/update_web_app_char_lstm.py

echo "Done! The character-level LSTM model has been trained, evaluated, and deployed."
echo "You can now test it interactively with: $PYTHON scripts/interactive_test_char_lstm.py"
