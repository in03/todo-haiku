import os
import sys
from pathlib import Path

print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Parent directory: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

# Check if the data file exists
data_path = "../data/syllable_dictionary.csv"
abs_data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "syllable_dictionary.csv"))

print(f"\nRelative path: {data_path}")
print(f"Absolute path: {abs_data_path}")
print(f"Relative path exists: {os.path.exists(data_path)}")
print(f"Absolute path exists: {os.path.exists(abs_data_path)}")

# List files in data directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
print(f"\nFiles in data directory ({data_dir}):")
if os.path.exists(data_dir):
    for file in os.listdir(data_dir):
        print(f"  - {file}")
else:
    print("  Data directory not found!")
