# Character-Level LSTM Syllable Counter Model

This directory contains scripts for training and evaluating a character-level LSTM syllable counter model for the Todo Haiku application.

## Approach

The character-level LSTM model treats syllabification as a sequence labeling task using BIO tags:
- **B**: Beginning of syllable
- **I**: Inside syllable
- **O**: Outside (not used in our implementation)

This approach directly models the syllable boundaries within words, which is more accurate than trying to predict the total syllable count directly.

## Model Architecture

The model uses a bidirectional LSTM architecture:
- Character embeddings (64 dimensions)
- 2-layer bidirectional LSTM (128 hidden dimensions)
- Dropout (0.2) for regularization
- Linear output layer for BIO tag classification

## Training the Character-Level LSTM Model

The `train_char_lstm.py` script trains a character-level LSTM model for syllable detection.

### Usage

```bash
# Train with default parameters
.venv/Scripts/python.exe scripts/train_char_lstm.py

# Train with custom parameters
.venv/Scripts/python.exe scripts/train_char_lstm.py --batch_size 64 --epochs 30 --learning_rate 0.001 --embedding_dim 64 --hidden_dim 128 --n_layers 2 --dropout 0.2 --early_stopping_patience 5
```

### Parameters

- `--data_path`: Path to the syllable dictionary CSV file (default: "data/syllable_dictionary_cmu.csv")
- `--model_dir`: Directory to save the model (default: "models")
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 30)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--embedding_dim`: Dimension of character embeddings (default: 64)
- `--hidden_dim`: Dimension of LSTM hidden states (default: 128)
- `--n_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--device`: Device to train on ('cuda' or 'cpu') (default: auto-detect)
- `--early_stopping_patience`: Number of epochs to wait for improvement before stopping (default: 5)
- `--max_samples`: Maximum number of samples to use for debugging (default: None)

## Interactive Testing

The `interactive_test_char_lstm.py` script allows you to test the character-level LSTM model interactively.

### Usage

```bash
.venv/Scripts/python.exe scripts/interactive_test_char_lstm.py
```

### Special Commands

- `stats`: Show detailed statistics
- `clear`: Clear metrics history
- `test`: Run test on common words

## Converting to ONNX Format

The `convert_char_lstm_to_onnx.py` script converts the character-level LSTM model to ONNX format for use in the web application.

### Usage

```bash
.venv/Scripts/python.exe scripts/convert_char_lstm_to_onnx.py
```

### Parameters

- `--model_path`: Path to the PyTorch model checkpoint (default: "models/syllable_char_lstm.pt")
- `--output_dir`: Directory to save the ONNX model (default: "models")
- `--device`: Device to load the model on ('cuda' or 'cpu') (default: auto-detect)

## Updating the Web Application

The `update_web_app_char_lstm.py` script updates the web application to use the character-level LSTM model.

### Usage

```bash
.venv/Scripts/python.exe scripts/update_web_app_char_lstm.py
```

## Automation

The `train_and_deploy_char_lstm.bat` (Windows) and `train_and_deploy_char_lstm.sh` (Linux/macOS) scripts automate the entire process of training, evaluating, and deploying the character-level LSTM model.

### Usage

```bash
# On Windows
train_and_deploy_char_lstm.bat

# On Linux/macOS
./train_and_deploy_char_lstm.sh
```

## Integration with Todo Haiku

After training and deploying the model, you can use it in the Todo Haiku application by setting the syllable counter type to `CHAR_LSTM`:

```typescript
// In your component
import {
  initCharLstmModel,
  setSyllableCounterType,
  SyllableCounterType,
  countSyllables
} from '~/utils/syllable-counter-switch';

// Use the character-level LSTM model
setSyllableCounterType(SyllableCounterType.CHAR_LSTM);
```

## Advantages of the Character-Level LSTM Approach

1. **Direct Modeling of Syllable Boundaries**: By treating syllabification as a sequence labeling task, the model directly learns where syllable boundaries occur in words.

2. **Better Generalization**: Character-level models can generalize better to unseen words by learning patterns at the character level.

3. **More Accurate for Complex Words**: This approach is particularly effective for longer, more complex words where simple rule-based approaches often fail.

4. **No Systematic Bias**: Unlike the previous approach, this model doesn't have a systematic bias toward overestimating syllable counts.

5. **Interpretable Predictions**: The BIO tags make it possible to visualize exactly where the model thinks syllable boundaries occur, making debugging easier.
