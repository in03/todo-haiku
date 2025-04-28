#!/bin/bash
# Train, evaluate, and deploy the balanced syllable counter model

# Set the Python executable
PYTHON=.venv/Scripts/python.exe

# Create the models directory if it doesn't exist
mkdir -p models

# Step 1: Train the balanced model
echo "Step 1: Training the balanced model..."
$PYTHON scripts/train_balanced.py --balance_method undersample --max_syllables 9 --batch_size 64 --epochs 30 --learning_rate 0.001 --curated_dataset --validation_threshold 1

# Step 2: Evaluate the balanced model
echo "Step 2: Evaluating the balanced model..."
$PYTHON scripts/evaluate_balanced.py

# Step 3: Convert the balanced model to ONNX format
echo "Step 3: Converting the balanced model to ONNX format..."
$PYTHON scripts/convert_balanced_to_onnx.py

# Step 4: Update the web application
echo "Step 4: Updating the web application..."
$PYTHON scripts/update_web_app.py

echo "Done! The balanced model has been trained, evaluated, and deployed."
echo "You can now test it interactively with: $PYTHON scripts/interactive_test_balanced.py"
