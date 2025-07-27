@echo off
REM Big Phoney API Development Setup Script for Windows

echo 🚀 Setting up Big Phoney API for local development...

REM Check if UV is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ UV is not installed. Please install it first:
    echo    https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

REM Install dependencies
echo 📦 Installing dependencies with UV...
uv sync

REM Create a .env file for local development
echo ⚙️  Creating .env file for local development...
(
echo # Local development configuration
echo PORT=8000
echo HOST=0.0.0.0
echo RELOAD=true
) > .env

echo ✅ Development setup complete!
echo.
echo 🎯 Next steps:
echo 1. Start the development server:
echo    uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 2. Test the API:
echo    uv run python test.py
echo.
echo 3. Update your Elixir client endpoint to:
echo    http://localhost:8000
echo.
echo 4. Test the Elixir integration:
echo    mix big_phoney test
pause 