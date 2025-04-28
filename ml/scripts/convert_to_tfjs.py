import torch
import tensorflow as tf
import tensorflowjs as tfjs
from model import SyllableCounter  # Your PyTorch model

def convert_to_tfjs():
    # Load PyTorch model
    pytorch_model = SyllableCounter()
    pytorch_model.load_state_dict(torch.load('models/syllable_counter.pt'))
    pytorch_model.eval()

    # Create equivalent TF model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_length),
        tf.keras.layers.LSTM(hidden_size, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    # Convert weights
    # ... weight conversion logic ...

    # Save as TF.js format
    tfjs.converters.save_keras_model(tf_model, 'models/syllable_model')