# ONNX Models

This directory contains ONNX models for the Todo Haiku application.

## Syllable Counter Model

The syllable counter model is used to count syllables in words for haiku validation.

### Files

- `syllable_counter.onnx`: The ONNX model file
- `char_vocab.json`: Character vocabulary for the model
- `model_metadata.json`: Metadata for the model

### How to Generate

The model is trained and converted to ONNX format using the scripts in the `ml` directory:

1. Download the CMU Pronouncing Dictionary:
   ```
   cd ml
   python scripts/download_cmudict.py
   ```

2. Train the model:
   ```
   python scripts/train.py
   ```

3. Convert to ONNX:
   ```
   python scripts/convert_to_onnx.py
   ```

4. Copy the generated files to this directory:
   ```
   cp ml/models/syllable_counter.onnx public/models/
   cp ml/models/char_vocab.json public/models/
   cp ml/models/model_metadata.json public/models/
   ```
