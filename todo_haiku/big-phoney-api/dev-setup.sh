#!/bin/bash

# Big Phoney API Development Setup Script

echo "🚀 Setting up Big Phoney API for local development..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install it first:"
    echo "   https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies with UV..."
uv sync

# Create a .env file for local development
echo "⚙️  Creating .env file for local development..."
cat > .env << EOF
# Local development configuration
PORT=8000
HOST=0.0.0.0
RELOAD=true
EOF

echo "✅ Development setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Start the development server:"
echo "   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Test the API:"
echo "   uv run python test.py"
echo ""
echo "3. Update your Elixir client endpoint to:"
echo "   http://localhost:8000"
echo ""
echo "4. Test the Elixir integration:"
echo "   mix big_phoney test" 