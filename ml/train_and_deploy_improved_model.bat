@echo off
REM Train, evaluate, and deploy the improved CMU-based syllable counter model

REM Set the Python executable
SET PYTHON=.venv\Scripts\python.exe

REM Create the models directory if it doesn't exist
if not exist models mkdir models

REM Step 0: Download the Kaggle dataset if it doesn't exist
echo Step 0: Downloading the Kaggle dataset if it doesn't exist...
if not exist data mkdir data
if not exist data\syllable_dictionary.csv %PYTHON% scripts\download_kaggle_dataset.py

REM Step 1: Create the CMU dataset
echo Step 1: Creating the CMU dataset...
%PYTHON% scripts/optimized_cmu_counter.py --mode create --input_path data/syllable_dictionary.csv --output_path data/syllable_dictionary_cmu.csv

REM Step 2: Train the improved model
echo Step 2: Training the improved model...
%PYTHON% scripts/train_cmu_improved.py --batch_size 64 --epochs 50 --learning_rate 0.001 --embedding_dim 128 --hidden_dim 256 --early_stopping_patience 10 --min_samples_per_class 500 --max_samples_per_class 5000 --bias_correction 4 --use_existing

REM Step 3: Convert the improved model to ONNX format
echo Step 3: Converting the improved model to ONNX format...
%PYTHON% scripts/convert_improved_to_onnx.py

REM Step 4: Update the web application
echo Step 4: Updating the web application...
%PYTHON% scripts/update_web_app_improved.py

echo Done! The improved model has been trained, evaluated, and deployed.
echo You can now test it interactively with: %PYTHON% scripts/interactive_test_improved.py
