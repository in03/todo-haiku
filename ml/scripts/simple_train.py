"""
A simplified version of the syllable counter training script.
This is similar to the user's original script but with better organization.
"""
# No need to import from syllable_counter package as this is a standalone script
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


class SimpleSyllableModel(nn.Module):
    """
    A simple LSTM-based model for syllable counting.
    """
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embeds = self.embedding(x)
        _, (hidden, _) = self.lstm(embeds)
        out = self.fc(hidden.squeeze(0))
        return out.squeeze(1)


def load_data(data_path=None):
    """
    Load data from a CSV file or use a mini dataset if no file is provided.
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        # Take a sample for quick training
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        data = {
            'word': df['word'].tolist(),
            'syllables': df['syllable_count'].tolist()
        }
    else:
        print("Using mini dataset")
        data = {
            'word': ['apple', 'banana', 'car', 'beautiful', 'interesting',
                     'different', 'experience', 'haiku', 'poetry', 'syllable'],
            'syllables': [2, 3, 1, 3, 4, 3, 4, 2, 3, 3]
        }

    return pd.DataFrame(data)


def train_simple_model(
    data_path=None,
    output_dir='../models',
    epochs=200,
    learning_rate=0.01,
    embed_dim=16,
    hidden_dim=32
):
    """
    Train a simple syllable counter model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Create vocabulary
    vocab = sorted(set(''.join(df['word'])))
    char_to_idx = {ch: idx+1 for idx, ch in enumerate(vocab)}  # Start from 1
    vocab_size = len(char_to_idx) + 1  # plus padding idx=0

    # Save vocabulary as JSON for JavaScript
    import json
    vocab_json_path = os.path.join(output_dir, 'simple_char_vocab.json')
    with open(vocab_json_path, 'w') as f:
        json.dump(char_to_idx, f)

    print(f"Saved vocabulary to {vocab_json_path}")

    # Preprocess data
    def word_to_tensor(word):
        indices = [char_to_idx.get(ch, 0) for ch in word]
        return torch.tensor(indices, dtype=torch.long)

    X = [word_to_tensor(word) for word in df['word']]
    y = torch.tensor(df['syllables'], dtype=torch.float32)

    # Pad inputs
    X = pad_sequence(X, batch_first=True)

    # Create model
    model = SimpleSyllableModel(vocab_size, embed_dim, hidden_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

    # Save model as PyTorch file
    model_path = os.path.join(output_dir, 'simple_syllable_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch model to {model_path}")

    # Save model as ONNX
    model.eval()
    dummy_input = torch.zeros((1, X.shape[1]), dtype=torch.long)
    onnx_path = os.path.join(output_dir, 'simple_syllable_model.onnx')

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    print(f"Exported ONNX model to {onnx_path}")

    # Create a metadata file with model information
    metadata = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_dim': hidden_dim,
        'max_word_len': X.shape[1]
    }

    metadata_path = os.path.join(output_dir, 'simple_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to {metadata_path}")

    print("✅ Model trained and exported successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple syllable counter model")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the dataset CSV file (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=16,
        help="Dimension of character embeddings"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Dimension of LSTM hidden states"
    )

    args = parser.parse_args()
    train_simple_model(**vars(args))
