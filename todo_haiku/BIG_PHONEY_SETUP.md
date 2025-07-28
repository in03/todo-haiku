# Big Phoney Microservice Integration

This guide explains how to set up and use the big-phoney Python microservice with your Elixir LiveView app for syllable counting.

## Overview

The setup consists of:
1. **Python FastAPI Microservice** (`big-phoney-api/`) - Hosts the big-phoney ML library
2. **Elixir Client Module** (`lib/todo_haiku/big_phoney_client.ex`) - Integrates with the microservice
3. **Simplified Syllable Counter** (`lib/todo_haiku/syllable_counter.ex`) - Clean interface to the microservice

## Quick Start

### 1. Deploy the Microservice

```bash
# Navigate to the microservice directory
cd big-phoney-api

# Make the deployment script executable
chmod +x deploy.sh

# Deploy to Fly.io
./deploy.sh
```

### 2. Configure the Endpoint

After deployment, set the actual endpoint URL:

```bash
# Set the environment variable (replace with your actual URL)
export BIG_PHONEY_ENDPOINT=https://your-actual-app-name.fly.dev

# Or add it to your Fly.io app secrets
flyctl secrets set BIG_PHONEY_ENDPOINT=https://your-actual-app-name.fly.dev
```

### 3. Test the Integration

```bash
# Test the microservice connection
mix big_phoney test

# Check microservice health
mix big_phoney health

# Test syllable counting
mix big_phoney count "hello"

# Compare microservice vs local implementation
mix big_phoney compare
```

### 3. Use in Your App

The `TodoHaiku.BigPhoneyClient` module provides several functions:

```elixir
# Simple syllable counting
{:ok, syllables} = TodoHaiku.BigPhoneyClient.count_syllables("hello")

# Detailed counting with word breakdown
{:ok, result} = TodoHaiku.BigPhoneyClient.count_syllables_detailed("hello world")
# Returns: %{syllables: 3, words: [%{word: "hello", syllables: 2}, %{word: "world", syllables: 1}]}

# Batch processing
word_counts = TodoHaiku.BigPhoneyClient.count_syllables_batch(["hello", "world", "beautiful"])
```

## Architecture

### Microservice (Python/FastAPI)
- **Location**: `big-phoney-api/`
- **Framework**: FastAPI with Uvicorn
- **ML Library**: big-phoney
- **Deployment**: Fly.io (256MB RAM, shared CPU)
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /syllables/simple` - Simple syllable count
  - `POST /syllables` - Detailed syllable breakdown

### Elixir Integration
- **Client Module**: `TodoHaiku.BigPhoneyClient`
- **Syllable Counter**: `TodoHaiku.SyllableCounter` - Clean interface to the microservice
- **HTTP Client**: Uses `req` library (already in your deps)
- **Error Handling**: Comprehensive error handling with timeouts

## Configuration

### Environment Variables

The microservice uses these environment variables:
- `PORT` - Server port (default: 8000)
- `PHX_HOST` - Your main app's hostname

### Fly.io Configuration

The microservice is configured for minimal cost:
- **Memory**: 256MB (free tier)
- **CPU**: Shared (free tier)
- **Auto-scaling**: Disabled (stays running)
- **Region**: Sydney (matches your main app)

## API Endpoints

### Simple Syllable Count
```bash
POST https://your-app-name.fly.dev/syllables/simple
Content-Type: application/json

{
  "text": "hello world"
}
```

Response:
```json
{
  "syllables": 3
}
```

### Detailed Syllable Breakdown
```bash
POST https://your-app-name.fly.dev/syllables
Content-Type: application/json

{
  "text": "hello world"
}
```

Response:
```json
{
  "text": "hello world",
  "syllables": 3,
  "words": [
    {"word": "hello", "syllables": 2},
    {"word": "world", "syllables": 1}
  ]
}
```

## Integration

The syllable counting is now fully integrated with the big-phoney microservice:

```elixir
# Use the simplified syllable counter
syllables = TodoHaiku.SyllableCounter.count_syllables_in_line("hello world")

# Or use the client directly
{:ok, syllables} = TodoHaiku.BigPhoneyClient.count_syllables("hello world")
```

The `TodoHaiku.SyllableCounter` module provides a clean interface that your existing code can use without changes.

## Monitoring and Maintenance

### Health Checks
```bash
# Check microservice health
mix big_phoney health

# Or directly
curl https://your-app-name.fly.dev/health
```

### Logs
```bash
# View microservice logs
flyctl logs -a big-phoney-api

# View your main app logs
flyctl logs -a todo-haiku
```

### Scaling
The microservice is configured for minimal cost but can be scaled if needed:

```bash
# Scale up (if needed)
flyctl scale count 2 -a big-phoney-api

# Scale down
flyctl scale count 1 -a big-phoney-api
```

## Cost Analysis

### Current Configuration (Minimal Cost)
- **Microservice**: 256MB RAM, shared CPU = Free tier
- **Main App**: 512MB RAM, shared CPU = Free tier
- **Total**: ~$0/month (within Fly.io free tier limits)

### If You Need More Performance
- **Microservice**: 512MB RAM = ~$1.94/month
- **Main App**: 1GB RAM = ~$3.88/month
- **Total**: ~$5.82/month

## Troubleshooting

### Microservice Not Responding
1. Check if it's deployed: `flyctl status -a big-phoney-api`
2. Check logs: `flyctl logs -a big-phoney-api`
3. Redeploy: `cd big-phoney-api && flyctl deploy`

### Integration Issues
1. Test connection: `mix big_phoney test`
2. Check health: `mix big_phoney health`
3. Verify endpoint URL in `TodoHaiku.BigPhoneyClient.endpoint/0`

### Performance Issues
1. Check response times: `mix big_phoney count "sophisticated"`
2. Monitor logs for timeouts
3. Consider increasing timeout values in the client

## Development

### Local Development
```bash
# Start the microservice locally
cd big-phoney-api
pip install -r requirements.txt
uvicorn main:app --reload

# Test locally
python test.py

# Update the endpoint in the Elixir client for local testing
# Change @endpoint to "http://localhost:8000" in big_phoney_client.ex
```

### Testing
```bash
# Test the microservice
cd big-phoney-api && python test.py

# Test the Elixir integration
mix big_phoney test
mix big_phoney compare
```

## Benefits

1. **Better Accuracy**: big-phoney is ML-based and more accurate than rule-based approaches
2. **Scalability**: Can handle high-frequency requests efficiently
3. **Simplicity**: Clean, focused codebase without complex fallback logic
4. **Cost-Effective**: Minimal additional cost on Fly.io
5. **Maintainable**: Clean separation between Elixir and Python code

## Next Steps

1. Deploy the microservice using the provided script
2. Test the integration with the Mix tasks
3. The syllable counting is already integrated into your haiku validation logic
4. Monitor performance and adjust as needed
5. Consider adding caching if you have high-frequency requests

The setup provides a clean, cost-effective solution for integrating the big-phoney library into your Elixir LiveView app. 