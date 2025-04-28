"""
Download the Kaggle English Phonetic and Syllable Count Dictionary.

Note: This script requires the Kaggle API credentials to be set up.
You need to have a Kaggle account and download your API token from
your account settings page.
"""
import argparse
import os
import subprocess
import zipfile
from pathlib import Path


def download_kaggle_dataset(output_dir: str):
    """
    Download the Kaggle English Phonetic and Syllable Count Dictionary.
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset information
    dataset_name = "schwartstack/english-phonetic-and-syllable-count-dictionary"
    
    # Check if kaggle command is available
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Kaggle CLI not found or not working properly.")
        print("Please install it with: pip install kaggle")
        print("And set up your API credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        return None
    
    # Download the dataset
    zip_path = os.path.join(output_dir, "syllable-dictionary.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Kaggle dataset to {zip_path}...")
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset_name, "-p", output_dir, "-o"],
                check=True
            )
            print("Download complete!")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading dataset: {e}")
            return None
    else:
        print(f"Dataset zip already exists at {zip_path}")
    
    # Extract the dataset
    csv_path = os.path.join(output_dir, "syllable_dictionary.csv")
    if not os.path.exists(csv_path):
        print(f"Extracting dataset to {output_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete!")
    else:
        print(f"Dataset already extracted to {csv_path}")
    
    return csv_path


def manual_download_instructions():
    """
    Print instructions for manually downloading the dataset.
    """
    print("\nAlternative: Manual Download Instructions")
    print("1. Go to https://www.kaggle.com/datasets/schwartstack/english-phonetic-and-syllable-count-dictionary")
    print("2. Click 'Download' (you need a Kaggle account)")
    print("3. Extract the downloaded zip file")
    print("4. Place the 'syllable_dictionary.csv' file in the 'ml/data' directory")
    print("5. Run the training script with: python scripts/train_kaggle.py --data_path data/syllable_dictionary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle Syllable Dictionary")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data",
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    csv_path = download_kaggle_dataset(args.output_dir)
    
    if csv_path is None:
        manual_download_instructions()
    else:
        print(f"\nDataset ready at: {csv_path}")
        print("You can now train the model with:")
        print(f"python scripts/train_kaggle.py --data_path {csv_path}")
