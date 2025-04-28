# Syllable Counter ML Model

This directory contains the machine learning model for syllable counting used in Todo Haiku.

## Overview

The syllable counter is trained on English text data and exported to ONNX format for use in the web application. It provides more accurate syllable counting than rule-based approaches, especially for edge cases and uncommon words.

## Directory Structure

- `data/`: Training and test datasets
- `models/`: Trained models (ONNX format)
- `notebooks/`: Jupyter notebooks for exploration
- `scripts/`: Training and conversion scripts
- `src/`: Python source code
  - `preprocessing/`: Data preprocessing utilities
  - `training/`: Model training code
  - `evaluation/`: Model evaluation code
- `tests/`: Unit tests for ML code

## Getting Started

### Setup with uv

```bash
# Install uv if you don't have it
# Using pip
pip install uv

# Or using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies (without ONNX)
cd ml
uv venv
uv pip install -e .
```

#### Handling ONNX Installation Issues

ONNX can be challenging to install, especially on Windows. Here are several options:

**Option 1: Install without ONNX first, then add it**
```bash
# Install core dependencies first
uv pip install -e .

# Then add ONNX dependencies
uv pip install -e ".[onnx]"
```

**Option 2: Use the setup script**
```bash
# Create and activate a virtual environment first
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Run the setup script
python scripts/setup_env.py
```

**Option 3: Install from requirements.txt**
```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install from requirements.txt
pip install -r requirements.txt
```

**Option 4: Skip ONNX for development**
If you're just developing or training models and don't need ONNX conversion yet:
```bash
# Install without ONNX
uv pip install -e .

# Train models without ONNX conversion
python scripts/simple_train.py
```

### Download the Kaggle Dataset

```bash
# Option 1: Using the Kaggle API (requires Kaggle account and API setup)
python scripts/download_kaggle_dataset.py

# Option 2: Manual download
# 1. Go to https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary
# 2. Download the dataset
# 3. Extract and place the CSV file in the data directory
```

### Training Options

#### 1. Simple Model (Quick Training)

```bash
# Train a simple model (similar to the original script)
python scripts/simple_train.py

# Use the Kaggle dataset with the simple model
python scripts/simple_train.py --data_path data/syllable_dictionary.csv
```

#### 2. Comprehensive Model (Better Accuracy)

```bash
# Train using the Kaggle dataset
python scripts/train_kaggle.py

# Convert to ONNX
python scripts/convert_to_onnx.py
```

### Evaluating the Model

```bash
python scripts/evaluate.py
```

## Integration with Todo Haiku

The ONNX model is loaded in the web application using ONNX Runtime Web. See `src/utils/syllable-counter-onnx.ts` for implementation details.

## Comparing Simple vs. Comprehensive Models

The repository includes two approaches:

1. **Simple Model** (`scripts/simple_train.py`):
   - Lightweight LSTM model
   - Direct regression output
   - Minimal preprocessing
   - Quick to train
   - Similar to the original script

2. **Comprehensive Model** (`scripts/train_kaggle.py`):
   - Bidirectional LSTM with multiple layers
   - Classification approach (0-9 syllables)
   - Feature engineering (vowel counts, special patterns)
   - Higher accuracy
   - More robust evaluation

## Step-by-Step Guide to Training Your Own Syllable Counter

This section provides detailed instructions for training your own syllable counter model from scratch.

### Prerequisites

- Python 3.9 or higher
- UV package manager (recommended) or pip
- Kaggle account (for downloading the dataset)

### 1. Set Up the Environment

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/in03/todo-haiku.git
cd todo-haiku

# Navigate to the ML directory
cd ml

# Create a virtual environment and install dependencies
uv venv
uv pip install -e .

# Alternative with standard Python tools
# python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
# pip install -e .
```

### 2. Download the Dataset

#### Option A: Using the Kaggle API

```bash
# Install the Kaggle CLI if you don't have it
pip install kaggle

# Set up your Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
# 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json (on macOS/Linux)

# Download the dataset
python scripts/download_kaggle_dataset.py
```

#### Option B: Manual Download

1. Go to [Kaggle English Phonetic and Syllable Count Dictionary](https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary)
2. Click "Download" (requires a Kaggle account)
3. Extract the downloaded zip file
4. Place the `syllable_dictionary.csv` file in the `ml/data` directory

### 3. Train the Model

#### Option A: Simple Model (Quick Training)

```bash
# Train with the mini dataset (for testing)
python scripts/simple_train.py

# Train with the Kaggle dataset
python scripts/simple_train.py --data_path data/syllable_dictionary.csv

# Additional options
python scripts/simple_train.py --data_path data/syllable_dictionary.csv --epochs 300 --learning_rate 0.005
```

#### Option B: Comprehensive Model (Better Accuracy)

```bash
# Train with default parameters
python scripts/train_kaggle.py

# With custom parameters
python scripts/train_kaggle.py --batch_size 128 --epochs 30 --learning_rate 0.0005

# Convert the trained model to ONNX format
python scripts/convert_to_onnx.py
```

### 4. Evaluate the Model

```bash
# Evaluate the comprehensive model
python scripts/evaluate.py

# With custom parameters
python scripts/evaluate.py --model_path models/syllable_counter.pt --data_path data/syllable_dictionary.csv
```

### 5. Use the Model in the Todo Haiku App

1. Copy the ONNX model files to the public directory:

```bash
# Create the models directory if it doesn't exist
mkdir -p ../public/models

# Copy the model files
cp models/syllable_counter.onnx ../public/models/
cp models/char_vocab.json ../public/models/
cp models/model_metadata.json ../public/models/

# For the simple model
cp models/simple_syllable_model.onnx ../public/models/
cp models/simple_char_vocab.json ../public/models/
cp models/simple_model_metadata.json ../public/models/
```

2. Update the app to use the ONNX model:

```typescript
// In your component
import {
  initOnnxModel,
  setSyllableCounterType,
  SyllableCounterType,
  countSyllables
} from '~/utils/syllable-counter-switch';

// Initialize the ONNX model
await initOnnxModel();

// Switch to ONNX-based syllable counting
setSyllableCounterType(SyllableCounterType.ONNX);

// Count syllables
const syllables = await countSyllables('hello world');
```

### 6. Troubleshooting

#### Common Issues

- **Model not loading**: Check that the model files are in the correct location and have the correct names
- **Training errors**: Ensure you have enough RAM for training (at least 8GB recommended)
- **Import errors**: Make sure you've installed the package with `uv pip install -e .`
- **Kaggle API errors**: Verify your API credentials are set up correctly

#### ONNX Installation Issues

ONNX can be challenging to install, especially on Windows. If you encounter errors:

1. **Try installing without ONNX first**:
   ```bash
   uv pip install -e .
   ```

   Then add ONNX separately:
   ```bash
   uv pip install -e ".[onnx]"
   ```

2. **Use specific versions**:
   ```bash
   pip install protobuf==3.20.3
   pip install onnx==1.14.1
   pip install onnxruntime==1.15.1
   ```

3. **Skip ONNX conversion**:
   You can still train models without converting to ONNX. The conversion step is only needed for deployment.

4. **Use the setup script**:
   ```bash
   python scripts/setup_env.py
   ```

5. **Check system requirements**:
   - Windows: Visual C++ build tools might be required
     - `scoop install cmake`
   - macOS: Xcode command line tools might be required
   - Linux: Build essentials and Python dev packages might be needed
