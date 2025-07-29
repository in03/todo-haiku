# Syllable API

A FastAPI microservice that provides syllable counting functionality using an ONNX neural network model with CMU dictionary fallback.

## Features

- Fast syllable counting using ONNX model inference
- CMU pronunciation dictionary for common words (instant lookup)
- FastAPI with async support
- Docker containerization
- Fly.io deployment ready
- Health check endpoint
- Detailed word-by-word syllable breakdown
- UV-compatible development setup

## Architecture

The service uses a two-tier approach for optimal performance:

1. **Dictionary Lookup** - CMU pronunciation dictionary for common English words (ultra-fast)
2. **ONNX Model** - Neural network for novel/unknown words (fast ML inference)

This hybrid approach provides both speed and accuracy, handling both common words and novel terms effectively.

## Local Development with UV

### Prerequisites
- [UV](https://docs.astral.sh/uv/) - Fast Python package manager
- Python 3.11+

### Setup
```bash
# Install dependencies with UV
uv sync

# Run the development server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
uv run python main.py
```

### Testing
```bash
# Run the test suite
uv run python test.py

# Test individual components
uv run python onnx_syllable_counter.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Simple Syllable Count
```bash
POST /syllables/simple
Content-Type: application/json

{
  "text": "hello world"
}

# Response
{
  "syllables": 3
}
```

### Detailed Syllable Analysis
```bash
POST /syllables
Content-Type: application/json

{
  "text": "beautiful programming language"
}

# Response
{
  "text": "beautiful programming language",
  "syllables": 8,
  "words": [
    {"word": "beautiful", "syllables": 3},
    {"word": "programming", "syllables": 3},
    {"word": "language", "syllables": 2}
  ]
}
```

### Haiku Analysis
```bash
POST /syllables/haiku
Content-Type: application/json

{
  "text": "Cherry blossoms fall\nSoftly on peaceful earth\nSpring has come again"
}

# Response
{
  "text": "Cherry blossoms fall\nSoftly on peaceful earth\nSpring has come again",
  "lines": [
    {"line": "Cherry blossoms fall", "syllables": 5},
    {"line": "Softly on peaceful earth", "syllables": 6},
    {"line": "Spring has come again", "syllables": 5}
  ]
}
```

## Docker Deployment

### Build and Run Locally
```bash
# Build the image
docker build -t syllable-api .

# Run the container
docker run -p 8000:8000 syllable-api
```

### Environment Variables
- `PORT` - Server port (default: 8000)

## Fly.io Deployment

The service is configured for easy deployment to Fly.io:

```bash
# Deploy to Fly.io
fly deploy

# Check status
fly status

# View logs
fly logs
```

## Performance

- **Initialization**: ~0.03 seconds
- **Dictionary words**: >10,000 words/second (instant lookups)
- **Novel words (ONNX)**: ~2,000 words/second (fast ML inference)
- **Memory usage**: ~50MB (lightweight ONNX runtime)

## Technical Details

### ONNX Model
- **Architecture**: Bidirectional GRU + Dense layers
- **Input**: One-hot encoded character sequences (max 18 chars)
- **Output**: Continuous syllable count (rounded to nearest integer)
- **Accuracy**: 95.82% on CMU test set
- **Size**: ~32KB (very compact)

### CMU Dictionary
- **Coverage**: ~134,000 English words
- **Format**: CMU pronunciation dictionary format
- **Lookup**: O(1) hash table lookup
- **Fallback**: ONNX model for unknown words

## Development

### Project Structure
```
syllable-api/
├── main.py                    # FastAPI application
├── onnx_syllable_counter.py   # Core syllable counting logic
├── syllable_model.onnx        # ONNX neural network model
├── model_metadata.json        # Model configuration
├── cmudict/                   # CMU pronunciation dictionary
├── test.py                    # Test suite
├── pyproject.toml            # UV project configuration
├── Dockerfile                # Container configuration
└── fly.toml                  # Fly.io deployment config
```

### Adding New Features
1. Extend the `SyllableService` class in `onnx_syllable_counter.py`
2. Add new endpoints in `main.py`
3. Update tests in `test.py`
4. Update this README

## License

MIT License - feel free to use this in your own projects! 