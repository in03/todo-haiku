@echo off
REM Syllable API Deployment Script for Windows
REM This script deploys the syllable microservice to Fly.io

echo 🚀 Deploying Syllable API to Fly.io...

REM Check if flyctl is installed
where flyctl >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ flyctl is not installed. Please install it first:
    echo    https://fly.io/docs/hands-on/install-flyctl/
    pause
    exit /b 1
)

REM Check if we're logged in to Fly.io
flyctl auth whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Not logged in to Fly.io. Please run: flyctl auth login
    pause
    exit /b 1
)

REM Create the app if it doesn't exist
echo 📋 Creating Fly.io app...
flyctl apps create syllable-api --org personal 2>nul || echo App already exists

REM Deploy the application
echo 🚀 Deploying application...
flyctl deploy

if %errorlevel% equ 0 (
    echo ✅ Deployment successful!
    echo.
    echo 🌐 Your Syllable API is now available at:
    echo    https://syllable-api.fly.dev
    echo.
    echo 🏥 Health check:
    echo    https://syllable-api.fly.dev/health
    echo.
    echo 📚 API documentation:
    echo    https://syllable-api.fly.dev/docs
    echo.
    echo 📊 Check deployment status:
    echo    flyctl status -a syllable-api
    echo.
    echo 📝 View logs:
    echo    flyctl logs -a syllable-api
) else (
    echo ❌ Deployment failed!
    echo 📝 Check logs: flyctl logs -a syllable-api
    pause
    exit /b 1
)

pause 