#!/bin/bash

# Big Phoney API Deployment Script
# This script deploys the big-phoney microservice to Fly.io

set -e

echo "🚀 Deploying Big Phoney API to Fly.io..."

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
flyctl apps create big-phoney-api --org personal || echo "App already exists"

# Deploy the application
echo "🚀 Deploying application..."
flyctl deploy

# Check the deployment status
echo "📊 Checking deployment status..."
flyctl status

echo "✅ Deployment complete!"

# Get the app URL
APP_HOSTNAME=$(flyctl info --json | grep '"hostname"' | cut -d'"' -f4)

echo "🌐 Your API is available at: https://$APP_HOSTNAME"
echo "📖 API documentation: https://$APP_HOSTNAME/docs"
echo "💚 Health check: https://$APP_HOSTNAME/health"

# Test the deployment
echo "🧪 Testing the deployment..."
sleep 5  # Wait a moment for the app to fully start

if curl -f https://$APP_HOSTNAME/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "⚠️  Health check failed. The app might still be starting up."
fi

echo ""
echo "🎉 Big Phoney API is now deployed and ready to use!" 