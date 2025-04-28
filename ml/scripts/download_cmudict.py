"""
Download the CMU Pronouncing Dictionary.
"""
import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


def download_cmudict(output_dir: str):
    """
    Download the CMU Pronouncing Dictionary.
    
    Args:
        output_dir: Directory to save the dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for the CMU Pronouncing Dictionary
    cmudict_url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    
    # Download the dictionary
    dict_path = os.path.join(output_dir, "cmudict.dict")
    if not os.path.exists(dict_path):
        print(f"Downloading CMU Pronouncing Dictionary to {dict_path}...")
        urllib.request.urlretrieve(cmudict_url, dict_path)
        print("Download complete!")
    else:
        print(f"CMU Pronouncing Dictionary already exists at {dict_path}")
    
    return dict_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CMU Pronouncing Dictionary")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data",
        help="Directory to save the dictionary"
    )
    
    args = parser.parse_args()
    download_cmudict(args.output_dir)
