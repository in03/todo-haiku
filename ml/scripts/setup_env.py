"""
Script to set up the Python environment for the syllable counter.
This script installs dependencies in a specific order to avoid issues.
"""
import os
import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a shell command and print the output."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=False)
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
    return process.returncode


def setup_environment():
    """Set up the Python environment for the syllable counter."""
    # Ensure pip is up to date
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install core dependencies first
    core_deps = [
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.2.2",
    ]
    run_command(f"{sys.executable} -m pip install {' '.join(core_deps)}")
    
    # Install PyTorch separately
    run_command(f"{sys.executable} -m pip install torch==2.0.1")
    
    # Install ONNX dependencies
    onnx_deps = [
        "protobuf==3.20.3",  # Specific version that works with ONNX
        "onnx==1.14.1",
    ]
    run_command(f"{sys.executable} -m pip install {' '.join(onnx_deps)}")
    
    # Install ONNX Runtime
    if sys.platform == "win32":
        # Windows-specific installation
        run_command(f"{sys.executable} -m pip install onnxruntime==1.15.1")
    else:
        # Linux/macOS installation
        run_command(f"{sys.executable} -m pip install onnxruntime==1.15.1")
    
    # Install remaining dependencies
    other_deps = [
        "transformers==4.30.2",
        "jupyter==1.0.0",
        "matplotlib==3.7.2",
        "tqdm==4.65.0",
    ]
    run_command(f"{sys.executable} -m pip install {' '.join(other_deps)}")
    
    # Install development dependencies
    dev_deps = [
        "pytest==7.3.1",
        "black==23.3.0",
        "isort==5.12.0",
        "mypy==1.3.0",
        "ruff==0.0.270",
    ]
    run_command(f"{sys.executable} -m pip install {' '.join(dev_deps)}")
    
    # Install the package in development mode
    run_command(f"{sys.executable} -m pip install -e .")
    
    print("\nEnvironment setup complete!")
    print("You can now run the scripts to train and evaluate the syllable counter.")


if __name__ == "__main__":
    setup_environment()
