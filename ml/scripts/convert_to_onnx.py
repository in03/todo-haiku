"""
Convert trained PyTorch model to ONNX format.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Check if ONNX is installed
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: ONNX is not installed. You can install it with:")
    print("  pip install onnx onnxruntime")
    print("Continuing without ONNX support...")

from syllable_counter.training import SyllableCounter


class SyllableCounterONNX(nn.Module):
    """
    Wrapper for SyllableCounter model to make it compatible with ONNX export.
    This wrapper handles the preprocessing of input words.
    """
    def __init__(self, model, char_to_idx, max_word_len=20):
        super().__init__()
        self.model = model
        self.char_to_idx = char_to_idx
        self.max_word_len = max_word_len

    def forward(self, word_chars, word_length, num_vowels, num_consonants,
                vowel_sequences, ends_with_e, ends_with_le, ends_with_ed):
        """
        Forward pass for ONNX export.

        Args:
            word_chars: Character indices tensor of shape (batch_size, max_word_len)
            word_length: Word length tensor of shape (batch_size, 1)
            num_vowels: Number of vowels tensor of shape (batch_size, 1)
            num_consonants: Number of consonants tensor of shape (batch_size, 1)
            vowel_sequences: Number of vowel sequences tensor of shape (batch_size, 1)
            ends_with_e: Boolean tensor of shape (batch_size, 1)
            ends_with_le: Boolean tensor of shape (batch_size, 1)
            ends_with_ed: Boolean tensor of shape (batch_size, 1)

        Returns:
            Syllable count predictions
        """
        # Combine features
        features = torch.cat([
            word_length,
            num_vowels,
            num_consonants,
            vowel_sequences,
            ends_with_e,
            ends_with_le,
            ends_with_ed
        ], dim=1)

        # Forward pass through the model
        outputs = self.model(word_chars, features)

        return outputs


def convert_to_onnx(
    model_path: str,
    output_dir: str,
    max_word_len: int = 20,
    device: str = None
):
    """
    Convert trained PyTorch model to ONNX format.

    Args:
        model_path: Path to the trained PyTorch model
        output_dir: Directory to save the ONNX model
        max_word_len: Maximum word length for input
        device: Device to use for conversion
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = SyllableCounter(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_features=checkpoint['num_features']
    ).to(device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(model_path), 'char_vocab.pt')
    char_to_idx = torch.load(vocab_path)

    # Create ONNX wrapper
    onnx_model = SyllableCounterONNX(model, char_to_idx, max_word_len).to(device)

    # Create dummy inputs
    batch_size = 1
    dummy_word_chars = torch.zeros((batch_size, max_word_len), dtype=torch.long).to(device)
    dummy_word_length = torch.tensor([[5.0]], dtype=torch.float32).to(device)
    dummy_num_vowels = torch.tensor([[2.0]], dtype=torch.float32).to(device)
    dummy_num_consonants = torch.tensor([[3.0]], dtype=torch.float32).to(device)
    dummy_vowel_sequences = torch.tensor([[2.0]], dtype=torch.float32).to(device)
    dummy_ends_with_e = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    dummy_ends_with_le = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    dummy_ends_with_ed = torch.tensor([[0.0]], dtype=torch.float32).to(device)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'syllable_counter.onnx')

    torch.onnx.export(
        onnx_model,
        (
            dummy_word_chars,
            dummy_word_length,
            dummy_num_vowels,
            dummy_num_consonants,
            dummy_vowel_sequences,
            dummy_ends_with_e,
            dummy_ends_with_le,
            dummy_ends_with_ed
        ),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=[
            'word_chars',
            'word_length',
            'num_vowels',
            'num_consonants',
            'vowel_sequences',
            'ends_with_e',
            'ends_with_le',
            'ends_with_ed'
        ],
        output_names=['syllable_count'],
        dynamic_axes={
            'word_chars': {0: 'batch_size'},
            'word_length': {0: 'batch_size'},
            'num_vowels': {0: 'batch_size'},
            'num_consonants': {0: 'batch_size'},
            'vowel_sequences': {0: 'batch_size'},
            'ends_with_e': {0: 'batch_size'},
            'ends_with_le': {0: 'batch_size'},
            'ends_with_ed': {0: 'batch_size'},
            'syllable_count': {0: 'batch_size'}
        }
    )

    print(f"Exported ONNX model to {onnx_path}")

    # Save vocabulary as JSON for JavaScript
    import json
    vocab_json_path = os.path.join(output_dir, 'char_vocab.json')
    with open(vocab_json_path, 'w') as f:
        json.dump(char_to_idx, f)

    print(f"Saved vocabulary as JSON to {vocab_json_path}")

    # Create a metadata file with model information
    metadata = {
        'max_word_len': max_word_len,
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_features': checkpoint['num_features'],
        'test_accuracy': checkpoint['test_accuracy']
    }

    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/syllable_counter.pt",
        help="Path to the trained PyTorch model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models",
        help="Directory to save the ONNX model"
    )
    parser.add_argument(
        "--max_word_len",
        type=int,
        default=20,
        help="Maximum word length for input"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for conversion ('cuda' or 'cpu')"
    )

    args = parser.parse_args()

    if not ONNX_AVAILABLE:
        print("ERROR: ONNX is required for model conversion.")
        print("Please install ONNX with: pip install onnx onnxruntime")
        print("Or install the optional dependencies: pip install -e '.[onnx]'")
        sys.exit(1)

    convert_to_onnx(**vars(args))
