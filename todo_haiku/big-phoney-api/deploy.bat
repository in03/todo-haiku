@echo off
REM Big Phoney API Deployment Script for Windows
REM This script deploys the big-phoney microservice to Fly.io

echo 🚀 Deploying Big Phoney API to Fly.io...

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
flyctl apps create big-phoney-api --org personal 2>nul || echo App already exists

REM Deploy the application
echo 🚀 Deploying application...
flyctl deploy

REM Check the deployment status
echo 📊 Checking deployment status...
flyctl status

echo ✅ Deployment complete!

REM Get the app URL
for /f "tokens=*" %%i in ('flyctl info --json ^| findstr "hostname"') do set HOSTNAME_LINE=%%i
for /f "tokens=2 delims=:" %%i in ("%HOSTNAME_LINE%") do set APP_HOSTNAME=%%i
set APP_HOSTNAME=%APP_HOSTNAME:"=%
set APP_HOSTNAME=%APP_HOSTNAME:,=%
set APP_HOSTNAME=%APP_HOSTNAME: =%

echo 🌐 Your API is available at: https://%APP_HOSTNAME%
echo 📖 API documentation: https://%APP_HOSTNAME%/docs
echo 💚 Health check: https://%APP_HOSTNAME%/health

REM Test the deployment
echo 🧪 Testing the deployment...
timeout /t 5 /nobreak >nul

curl -f https://%APP_HOSTNAME%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Health check passed!
) else (
    echo ⚠️  Health check failed. The app might still be starting up.
)

echo.
echo 🎉 Big Phoney API is now deployed and ready to use!
pause 