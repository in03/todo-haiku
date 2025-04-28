"""
Analyze the syllable dictionary dataset to understand the distribution of syllable counts.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the dataset
data_path = Path("data/syllable_dictionary.csv")
df = pd.read_csv(data_path)

# Print basic information
print(f"Total words: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Analyze syllable count distribution
syl_counts = df['syl'].value_counts().sort_index()
print("\nSyllable count distribution:")
for syl, count in syl_counts.items():
    print(f"{syl} syllables: {count} words ({count/len(df):.2%})")

# Calculate statistics
print(f"\nMean syllables per word: {df['syl'].mean():.2f}")
print(f"Median syllables per word: {df['syl'].median():.2f}")
print(f"Min syllables: {df['syl'].min()}")
print(f"Max syllables: {df['syl'].max()}")

# Analyze word length vs syllable count
df['word_length'] = df['word'].apply(len)
print("\nWord length statistics:")
print(f"Mean word length: {df['word_length'].mean():.2f}")
print(f"Median word length: {df['word_length'].median():.2f}")
print(f"Min word length: {df['word_length'].min()}")
print(f"Max word length: {df['word_length'].max()}")

# Calculate correlation between word length and syllable count
correlation = df['word_length'].corr(df['syl'])
print(f"\nCorrelation between word length and syllable count: {correlation:.2f}")

# Group by syllable count and calculate average word length
avg_length_by_syl = df.groupby('syl')['word_length'].mean()
print("\nAverage word length by syllable count:")
for syl, avg_len in avg_length_by_syl.items():
    print(f"{syl} syllables: {avg_len:.2f} characters")

# Create a sample of words for each syllable count
print("\nSample words for each syllable count:")
for syl in sorted(df['syl'].unique()):
    sample_words = df[df['syl'] == syl]['word'].sample(min(5, len(df[df['syl'] == syl]))).tolist()
    print(f"{syl} syllables: {', '.join(sample_words)}")

# Save the plot
plt.figure(figsize=(10, 6))
plt.bar(syl_counts.index, syl_counts.values)
plt.xlabel('Number of Syllables')
plt.ylabel('Number of Words')
plt.title('Distribution of Syllable Counts in Dataset')
plt.xticks(np.arange(min(syl_counts.index), max(syl_counts.index)+1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('data/syllable_distribution.png')
print("\nSyllable distribution plot saved to data/syllable_distribution.png")
