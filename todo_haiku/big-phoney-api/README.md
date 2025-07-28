# Big Phoney API

A FastAPI microservice that provides syllable counting functionality using the big-phoney Python library.

## Features

- Count syllables in text using ML-based big-phoney library
- FastAPI with async support
- Docker containerization
- Fly.io deployment ready
- Health check endpoint
- Detailed word-by-word syllable breakdown
- UV-compatible development setup

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
# Run tests
uv run pytest

# Test the API manually
uv run python test.py
```

## Traditional Setup (pip)

If you prefer pip:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn main:app --reload
```

## API Endpoints

### GET /
Returns a simple status message.

### GET /health
Health check endpoint for monitoring.

### POST /syllables
Count syllables in text with detailed word breakdown.

**Request:**
```json
{
  "text": "hello world"
}
```

**Response:**
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

### POST /syllables/simple
Simple endpoint that returns just the total syllable count.

**Request:**
```json
{
  "text": "hello world"
}
```

**Response:**
```json
{
  "syllables": 3
}
```

## Docker Deployment

1. Build the image:
```bash
docker build -t big-phoney-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 big-phoney-api
```

## Fly.io Deployment

### Separate Deployment (Recommended)
The microservice is designed to be deployed separately from your main Elixir app:

1. Install Fly CLI if not already installed
2. Navigate to the big-phoney-api directory
3. Deploy:
```bash
# On Windows
deploy.bat

# On Unix/Linux
./deploy.sh
```

### Cost Analysis
- **Current Configuration**: 256MB RAM, shared CPU = **Free tier**
- **Main App**: 512MB RAM, shared CPU = **Free tier**
- **Total**: ~$0/month (within Fly.io free tier limits)

### Combined Deployment (Not Recommended)
While technically possible to deploy both services in one VM, it's not advisable because:
- **Complexity**: Managing two different runtime environments (Python + Elixir)
- **Resource Conflicts**: Both services competing for the same resources
- **Scaling Issues**: Can't scale services independently
- **Maintenance**: Harder to update/deploy services separately

## Integration with Elixir

The Elixir client is configured to use the deployed microservice by default. For local development:

1. Start the microservice locally:
```bash
cd big-phoney-api
uv run uvicorn main:app --reload
```

2. Update the endpoint in `lib/todo_haiku/big_phoney_client.ex`:
```elixir
@endpoint "http://localhost:8000"  # For local development
```

3. Test the integration:
```bash
mix big_phoney test
mix big_phoney health
```

## Mix Tasks

The project includes Mix tasks for testing the microservice:

```bash
# Test the microservice connection
mix big_phoney test

# Check microservice health
mix big_phoney health

# Count syllables in a word
mix big_phoney count "hello"

# Compare microservice vs local implementation
mix big_phoney compare
```

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

## Benefits of Separate Deployment

1. **Cost Efficiency**: Both services fit within Fly.io free tier
2. **Independent Scaling**: Scale services based on their specific needs
3. **Technology Isolation**: Each service uses its optimal runtime
4. **Maintenance**: Update/deploy services independently
5. **Reliability**: Service failures don't affect each other

## Troubleshooting

### Microservice Not Responding
1. Check if it's deployed: `flyctl status -a big-phoney-api`
2. Check logs: `flyctl logs -a big-phoney-api`
3. Redeploy: `cd big-phoney-api && flyctl deploy`

### Local Development Issues
1. Ensure UV is installed: `uv --version`
2. Check dependencies: `uv sync`
3. Verify the service is running: `curl http://localhost:8000/health`

### Integration Issues
1. Test connection: `mix big_phoney test`
2. Check health: `mix big_phoney health`
3. Verify endpoint URL in `TodoHaiku.BigPhoneyClient.endpoint/0` 