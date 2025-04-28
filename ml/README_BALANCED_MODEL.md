# Balanced Syllable Counter Model

This directory contains scripts for training and evaluating a balanced syllable counter model for the Todo Haiku application.

## Problem Statement

The original syllable counter model consistently overestimates syllable counts by approximately 2 syllables. This is likely due to two main issues:

1. **Imbalance in the training data**: Certain syllable counts are overrepresented, biasing the model.
2. **Incorrect syllable counts in the dataset**: Some words in the dataset have incorrect syllable counts, which confuses the model during training.

## Solution

We've created a set of scripts to train a more balanced and accurate model that should provide more accurate syllable counts across all word types, regardless of length or complexity. Our solution addresses both issues:

1. **Data Validation and Curation**: We validate the dataset using a rule-based syllable counter and manually correct known problematic words.
2. **Balanced Training**: We balance the dataset to ensure a more even distribution of syllable counts.

## Dataset Analysis

The Kaggle English Phonetic and Syllable Count Dictionary contains approximately 135,000 words with their syllable counts. The distribution is heavily skewed:

- 1 syllable: ~14% of words
- 2 syllables: ~48% of words
- 3 syllables: ~27% of words
- 4 syllables: ~9% of words
- 5+ syllables: ~3% of words

This imbalance likely contributes to the model's bias toward higher syllable counts.

## Data Validation and Curation

The `validate_dataset.py` script validates the syllable dictionary dataset by comparing with a rule-based syllable counter and creates a curated dataset with more accurate syllable counts.

### Usage

```bash
# Validate the dataset
.venv/Scripts/python.exe scripts/validate_dataset.py

# Validate with custom parameters
.venv/Scripts/python.exe scripts/validate_dataset.py --data_path data/syllable_dictionary.csv --threshold 1 --sample_size 2000
```

### Parameters

- `--data_path`: Path to the syllable dictionary CSV file (default: "data/syllable_dictionary.csv")
- `--output_path`: Path to save the validated dataset (default: "data/syllable_dictionary_validated.csv")
- `--curated_path`: Path to save the curated dataset (default: "data/syllable_dictionary_curated.csv")
- `--sample_size`: Number of words to sample for manual verification (default: 1000)
- `--threshold`: Maximum allowed difference between dataset and rule-based counts (default: 1)

## Training a Balanced Model

The `train_balanced.py` script creates a more balanced dataset by either undersampling or oversampling to ensure a more even distribution of syllable counts. It also includes data validation to filter out words with incorrect syllable counts.

### Usage

```bash
# Train with undersampling (default)
.venv/Scripts/python.exe scripts/train_balanced.py

# Train with oversampling
.venv/Scripts/python.exe scripts/train_balanced.py --balance_method oversample

# Train with curated dataset
.venv/Scripts/python.exe scripts/train_balanced.py --curated_dataset

# Train with custom parameters
.venv/Scripts/python.exe scripts/train_balanced.py --balance_method undersample --max_syllables 9 --batch_size 128 --epochs 50 --learning_rate 0.0005 --curated_dataset
```

### Parameters

- `--data_path`: Path to the syllable dictionary CSV file (default: "data/syllable_dictionary.csv")
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
- `--validate`: Validate the dataset using a rule-based syllable counter (default: False)
- `--validation_threshold`: Maximum allowed difference between dataset and rule-based counts (default: 1)
- `--curated_dataset`: Use a curated dataset with manually verified syllable counts (default: False)

## Evaluating the Model

The `evaluate_balanced.py` script compares the performance of the original and balanced models on a test set.

### Usage

```bash
.venv/Scripts/python.exe scripts/evaluate_balanced.py
```

### Parameters

- `--model_path`: Path to the original model checkpoint (default: "models/syllable_counter.pt")
- `--balanced_model_path`: Path to the balanced model checkpoint (default: "models/syllable_counter_balanced.pt")
- `--data_path`: Path to the test data CSV file (default: "data/syllable_dictionary.csv")
- `--output_dir`: Directory to save evaluation results (default: "models/evaluation")
- `--batch_size`: Batch size for evaluation (default: 64)
- `--device`: Device to evaluate on ('cuda' or 'cpu') (default: auto-detect)

## Converting to ONNX Format

The `convert_balanced_to_onnx.py` script converts the balanced model to ONNX format for use in the web application.

### Usage

```bash
.venv/Scripts/python.exe scripts/convert_balanced_to_onnx.py
```

### Parameters

- `--model_path`: Path to the PyTorch model checkpoint (default: "models/syllable_counter_balanced.pt")
- `--output_dir`: Directory to save the ONNX model (default: "models")
- `--device`: Device to load the model on ('cuda' or 'cpu') (default: auto-detect)

## Interactive Testing

The `interactive_test_balanced.py` script allows you to test the balanced model interactively.

### Usage

```bash
.venv/Scripts/python.exe scripts/interactive_test_balanced.py
```

## Integration with Todo Haiku

After training and converting the model to ONNX format, copy the model files to the public directory:

```bash
# Create the models directory if it doesn't exist
mkdir -p ../public/models

# Copy the model files
cp models/syllable_counter_balanced.onnx ../public/models/
cp models/char_vocab_balanced.json ../public/models/
cp models/model_metadata_balanced.json ../public/models/
```

Then update the app to use the balanced model:

```typescript
// In your component
import {
  initOnnxModel,
  setSyllableCounterType,
  SyllableCounterType,
  countSyllables
} from '~/utils/syllable-counter-switch';

// Use the balanced model
setSyllableCounterType(SyllableCounterType.BALANCED_ONNX);
```

## Expected Improvements

The balanced model should provide more accurate syllable counts across all word types, without the need for a bias correction factor. This will improve the user experience in the Todo Haiku application by providing more accurate feedback on haiku syllable counts.
