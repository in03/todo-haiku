# CMU-Based Syllable Counter Model

This directory contains scripts for training and evaluating a syllable counter model based on the CMU Pronouncing Dictionary for the Todo Haiku application.

## Problem Statement

The Kaggle English Phonetic and Syllable Count Dictionary used for the original model contains many incorrect syllable counts, which affects the model's accuracy. For example:

- "winebarger" is listed as 4 syllables (should be 3)
- "nickelodeon" is listed as 4 syllables (should be 5)
- "collaborationist" is listed as 6 syllables (should be 7)
- "authoritarianism" is listed as 7 syllables (should be 8)

## Solution

We've created a set of scripts to train a more accurate model using the CMU Pronouncing Dictionary, which is a well-established linguistic resource developed by Carnegie Mellon University and is widely used in speech recognition and natural language processing research.

## CMU Pronouncing Dictionary

The CMU Pronouncing Dictionary has several advantages over the Kaggle dataset:

1. **Linguistic Expertise**: Created by linguistic experts with phonetic transcriptions based on North American English pronunciation
2. **Accuracy**: Generally considered highly accurate for syllable counting because it's based on phonemes
3. **Consistency**: Uses a standardized phonetic notation system
4. **Research-Grade**: Used in academic and commercial speech systems
5. **Actively Maintained**: Has been maintained and updated over decades

## Creating a CMU-Based Dataset

The `cmu_syllable_counter.py` script provides several functions for working with the CMU Pronouncing Dictionary:

1. **Interactive Testing**: Test syllable counting interactively
2. **Dataset Creation**: Create a dataset with syllable counts from the CMU Pronouncing Dictionary
3. **Dataset Processing**: Process the Kaggle dataset and compare with CMU counts
4. **Test Dataset Creation**: Create a test dataset with words from the CMU dict

### Usage

```bash
# Interactive testing
.venv/Scripts/python.exe scripts/cmu_syllable_counter.py --mode interactive

# Create a CMU-based dataset
.venv/Scripts/python.exe scripts/cmu_syllable_counter.py --mode create --input_path data/syllable_dictionary.csv --output_path data/syllable_dictionary_cmu.csv

# Process the Kaggle dataset and compare with CMU counts
.venv/Scripts/python.exe scripts/cmu_syllable_counter.py --mode process --input_path data/syllable_dictionary.csv --output_path data/syllable_dictionary_processed.csv

# Create a test dataset with words from the CMU dict
.venv/Scripts/python.exe scripts/cmu_syllable_counter.py --mode test --output_path data/syllable_dictionary_test.csv --num_words 1000
```

## Training a CMU-Based Model

The `train_cmu.py` script trains a syllable counter model using a dataset created with the CMU Pronouncing Dictionary.

### Usage

```bash
# Train with undersampling (default)
.venv/Scripts/python.exe scripts/train_cmu.py

# Train with oversampling
.venv/Scripts/python.exe scripts/train_cmu.py --balance_method oversample

# Train with custom parameters
.venv/Scripts/python.exe scripts/train_cmu.py --balance_method undersample --max_syllables 9 --batch_size 128 --epochs 50 --learning_rate 0.0005
```

### Parameters

- `--data_path`: Path to the CMU dataset (default: "data/syllable_dictionary_cmu.csv")
- `--model_dir`: Directory to save the model (default: "models")
- `--balance_method`: Method to balance the dataset ('undersample', 'oversample', or 'weighted') (default: 'undersample')
- `--max_syllables`: Maximum number of syllables to include (default: 9)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 30)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--embedding_dim`: Dimension of character embeddings (default: 64)
- `--hidden_dim`: Dimension of LSTM hidden states (default: 128)
- `--device`: Device to train on ('cuda' or 'cpu') (default: auto-detect)
- `--early_stopping_patience`: Number of epochs to wait for improvement before stopping (default: 5)
- `--use_existing`: Use existing CMU dataset if available (default: False)

## Interactive Testing

The `interactive_test_cmu.py` script allows you to test the CMU-based model interactively.

### Usage

```bash
.venv/Scripts/python.exe scripts/interactive_test_cmu.py
```

## Converting to ONNX Format

The `convert_cmu_to_onnx.py` script converts the CMU-based model to ONNX format for use in the web application.

### Usage

```bash
.venv/Scripts/python.exe scripts/convert_cmu_to_onnx.py
```

### Parameters

- `--model_path`: Path to the PyTorch model checkpoint (default: "models/syllable_counter_cmu.pt")
- `--output_dir`: Directory to save the ONNX model (default: "models")
- `--device`: Device to load the model on ('cuda' or 'cpu') (default: auto-detect)

## Updating the Web Application

The `update_web_app_cmu.py` script updates the web application to use the CMU-based model.

### Usage

```bash
.venv/Scripts/python.exe scripts/update_web_app_cmu.py
```

## Automation

The `train_and_deploy_cmu_model.bat` (Windows) and `train_and_deploy_cmu_model.sh` (Linux/macOS) scripts automate the entire process of training, evaluating, and deploying the CMU-based model.

### Usage

```bash
# On Windows
train_and_deploy_cmu_model.bat

# On Linux/macOS
./train_and_deploy_cmu_model.sh
```

## Integration with Todo Haiku

After training and deploying the model, you can use it in the Todo Haiku application by setting the syllable counter type to `CMU_ONNX`:

```typescript
// In your component
import {
  initOnnxModel,
  setSyllableCounterType,
  SyllableCounterType,
  countSyllables
} from '~/utils/syllable-counter-switch';

// Use the CMU model
setSyllableCounterType(SyllableCounterType.CMU_ONNX);
```

## Expected Improvements

The CMU-based model should provide more accurate syllable counts across all word types, without the need for a bias correction factor. This will improve the user experience in the Todo Haiku application by providing more accurate feedback on haiku syllable counts.
