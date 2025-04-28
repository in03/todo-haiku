"""
Convert the CMU-based syllable counter model to ONNX format.
"""
import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to find syllable_counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syllable_counter.training import SyllableCounter


def convert_to_onnx(
    model_path,
    output_dir,
    device=None
):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        model_path: Path to the PyTorch model checkpoint
        output_dir: Directory to save the ONNX model
        device: Device to load the model on ('cuda' or 'cpu')
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
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
    vocab_path = os.path.join(os.path.dirname(model_path), 'char_vocab_cmu.json')
    with open(vocab_path, 'r') as f:
        char_to_idx = json.load(f)

    # Get max word length from the model
    # This is a bit of a hack, but we need to know the max word length for the ONNX model
    # We'll use the first batch from the training data to get this
    max_word_len = 20  # Default value

    # Create dummy inputs
    dummy_char_ids = torch.zeros((1, max_word_len), dtype=torch.long).to(device)
    dummy_features = torch.zeros((1, checkpoint['num_features']), dtype=torch.float32).to(device)

    # Export model to ONNX
    torch.onnx.export(
        model,
        (dummy_char_ids, dummy_features),
        os.path.join(output_dir, 'syllable_counter_cmu.onnx'),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['char_ids', 'word_length', 'num_vowels', 'num_consonants',
                    'vowel_sequences', 'ends_with_e', 'ends_with_le', 'ends_with_ed'],
        output_names=['syllable_count'],
        dynamic_axes={
            'char_ids': {0: 'batch_size'},
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

    print(f"Model exported to {os.path.join(output_dir, 'syllable_counter_cmu.onnx')}")

    # Save vocabulary as JSON
    with open(os.path.join(output_dir, 'char_vocab_cmu.json'), 'w') as f:
        json.dump(char_to_idx, f)

    print(f"Vocabulary saved to {os.path.join(output_dir, 'char_vocab_cmu.json')}")

    # Save model metadata
    metadata = {
        'max_word_len': max_word_len,
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'num_features': checkpoint['num_features'],
        'test_accuracy': checkpoint['test_accuracy']
    }

    with open(os.path.join(output_dir, 'model_metadata_cmu.json'), 'w') as f:
        json.dump(metadata, f)

    print(f"Metadata saved to {os.path.join(output_dir, 'model_metadata_cmu.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert syllable counter model to ONNX format")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/syllable_counter_cmu.pt",
        help="Path to the PyTorch model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the ONNX model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load the model on ('cuda' or 'cpu')"
    )

    args = parser.parse_args()
    convert_to_onnx(**vars(args))
