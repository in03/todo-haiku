#!/bin/bash

# Syllable API Deployment Script
# This script deploys the syllable microservice to Fly.io

set -e

echo "🚀 Deploying Syllable API to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "   https://fly.io/docs/hands-on/install-flyctl/"
    exit 1
fi

# Check if we're logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
fi

# Create the app if it doesn't exist
echo "📋 Creating Fly.io app..."
flyctl apps create syllable-api --org personal || echo "App already exists"

# Deploy the application
echo "🚀 Deploying application..."
flyctl deploy

if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🌐 Your Syllable API is now available at:"
    echo "   https://syllable-api.fly.dev"
    echo ""
    echo "🏥 Health check:"
    echo "   https://syllable-api.fly.dev/health"
    echo ""
    echo "📚 API documentation:"
    echo "   https://syllable-api.fly.dev/docs"
    echo ""
    echo "📊 Check deployment status:"
    echo "   flyctl status -a syllable-api"
    echo ""
    echo "📝 View logs:"
    echo "   flyctl logs -a syllable-api"
else
    echo "❌ Deployment failed!"
    echo "📝 Check logs: flyctl logs -a syllable-api"
    exit 1
fi 