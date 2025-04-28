# Improved CMU-Based Syllable Counter Model

This directory contains scripts for training and evaluating an improved syllable counter model based on the CMU Pronouncing Dictionary for the Todo Haiku application.

## Problem Statement

The original model had a systematic bias, consistently predicting too many syllables (e.g., predicting 7 syllables for "shoe" which should be 1 syllable). This was likely due to:

1. **Dataset Balancing Issue**: The aggressive balancing approach created an artificial distribution that didn't reflect the natural distribution of syllables in English words.
2. **Training Data Size**: With only 122 examples per syllable count, the model didn't have enough data to learn the complex patterns.
3. **Model Architecture**: The model architecture might not have been optimal for this task.

## Solution

We've created an improved training approach that addresses these issues:

1. **Weighted Loss Function**: Instead of artificially balancing the dataset by undersampling, we use a weighted loss function that gives more importance to underrepresented classes.
2. **Preserving Natural Distribution**: We keep more of the natural distribution while ensuring all classes have sufficient examples.
3. **Bias Correction**: We add a bias correction term to account for the model's tendency to overestimate.
4. **Larger Model**: We use a larger model with more parameters to better capture the complexity of syllable counting.
5. **More Training Data**: We ensure each syllable count has at least 500 examples.

## Training the Improved Model

The `train_cmu_improved.py` script trains a syllable counter model using the improved approach.

### Usage

```bash
# Train with default parameters
.venv/Scripts/python.exe scripts/train_cmu_improved.py

# Train with custom parameters
.venv/Scripts/python.exe scripts/train_cmu_improved.py --batch_size 64 --epochs 50 --learning_rate 0.001 --embedding_dim 128 --hidden_dim 256 --early_stopping_patience 10 --min_samples_per_class 500 --max_samples_per_class 5000 --bias_correction 2
```

### Parameters

- `--data_path`: Path to the CMU dataset (default: "data/syllable_dictionary_cmu.csv")
- `--model_dir`: Directory to save the model (default: "models")
- `--max_syllables`: Maximum number of syllables to include (default: 9)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--embedding_dim`: Dimension of character embeddings (default: 128)
- `--hidden_dim`: Dimension of LSTM hidden states (default: 256)
- `--device`: Device to train on ('cuda' or 'cpu') (default: auto-detect)
- `--early_stopping_patience`: Number of epochs to wait for improvement before stopping (default: 10)
- `--use_existing`: Use existing CMU dataset if available (default: False)
- `--min_samples_per_class`: Minimum number of samples per class (default: 500)
- `--max_samples_per_class`: Maximum number of samples per class (default: 5000)
- `--bias_correction`: Bias correction term to apply to predictions (default: 0)

## Interactive Testing

The `interactive_test_improved.py` script allows you to test the improved model interactively.

### Usage

```bash
.venv/Scripts/python.exe scripts/interactive_test_improved.py
```

### Special Commands

- `stats`: Show detailed statistics
- `clear`: Clear metrics history
- `test`: Run test on common words

## Converting to ONNX Format

The `convert_improved_to_onnx.py` script converts the improved model to ONNX format for use in the web application.

### Usage

```bash
.venv/Scripts/python.exe scripts/convert_improved_to_onnx.py
```

### Parameters

- `--model_path`: Path to the PyTorch model checkpoint (default: "models/syllable_counter_cmu_improved.pt")
- `--output_dir`: Directory to save the ONNX model (default: "models")
- `--device`: Device to load the model on ('cuda' or 'cpu') (default: auto-detect)

## Updating the Web Application

The `update_web_app_improved.py` script updates the web application to use the improved model.

### Usage

```bash
.venv/Scripts/python.exe scripts/update_web_app_improved.py
```

## Automation

The `train_and_deploy_improved_model.bat` (Windows) and `train_and_deploy_improved_model.sh` (Linux/macOS) scripts automate the entire process of training, evaluating, and deploying the improved model.

### Usage

```bash
# On Windows
train_and_deploy_improved_model.bat

# On Linux/macOS
./train_and_deploy_improved_model.sh
```

## Integration with Todo Haiku

After training and deploying the model, you can use it in the Todo Haiku application by setting the syllable counter type to `IMPROVED_ONNX`:

```typescript
// In your component
import {
  initOnnxModel,
  setSyllableCounterType,
  SyllableCounterType,
  countSyllables
} from '~/utils/syllable-counter-switch';

// Use the improved model
setSyllableCounterType(SyllableCounterType.IMPROVED_ONNX);
```

## Expected Improvements

The improved model should provide more accurate syllable counts across all word types, with the bias correction term addressing the systematic overestimation issue. This will improve the user experience in the Todo Haiku application by providing more accurate feedback on haiku syllable counts.
