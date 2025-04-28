"""
Convert the character-level LSTM syllable counter model to ONNX format.
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
from train_char_lstm import CharLSTM


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
    model = CharLSTM(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=3,  # B, I, O
        n_layers=checkpoint['n_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load vocabulary
    vocab_path = os.path.join(os.path.dirname(model_path), 'char_vocab_lstm.json')
    with open(vocab_path, 'r') as f:
        char_to_idx = json.load(f)
    
    # Create a wrapper class for ONNX export
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, chars, lengths):
            return self.model(chars, lengths)
    
    wrapper = ONNXWrapper(model)
    
    # Create dummy inputs
    max_word_len = 30  # Maximum word length
    dummy_chars = torch.zeros((1, max_word_len), dtype=torch.long).to(device)
    dummy_lengths = torch.tensor([max_word_len], dtype=torch.long).to(device)
    
    # Export model to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_chars, dummy_lengths),
        os.path.join(output_dir, 'syllable_char_lstm.onnx'),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['chars', 'lengths'],
        output_names=['bio_tags'],
        dynamic_axes={
            'chars': {0: 'batch_size', 1: 'seq_len'},
            'lengths': {0: 'batch_size'},
            'bio_tags': {0: 'batch_size', 1: 'seq_len'}
        }
    )
    
    print(f"Model exported to {os.path.join(output_dir, 'syllable_char_lstm.onnx')}")
    
    # Save vocabulary as JSON
    with open(os.path.join(output_dir, 'char_vocab_lstm.json'), 'w') as f:
        json.dump(char_to_idx, f)
    
    print(f"Vocabulary saved to {os.path.join(output_dir, 'char_vocab_lstm.json')}")
    
    # Save model metadata
    metadata = {
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'hidden_dim': checkpoint['hidden_dim'],
        'n_layers': checkpoint['n_layers'],
        'dropout': checkpoint['dropout'],
        'test_accuracy': checkpoint['test_syllable_accuracy']
    }
    
    with open(os.path.join(output_dir, 'model_metadata_char_lstm.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Metadata saved to {os.path.join(output_dir, 'model_metadata_char_lstm.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert syllable counter model to ONNX format")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/syllable_char_lstm.pt",
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
