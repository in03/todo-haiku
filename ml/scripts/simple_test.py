"""
Simple test script for the decision tree syllable counter.
"""
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def main():
    print("Testing decision tree classifier...")
    
    # Create a simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Train a decision tree
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    # Make predictions
    predictions = clf.predict(X)
    
    # Print results
    print("Predictions:", predictions)
    print("Accuracy:", np.mean(predictions == y))

if __name__ == "__main__":
    main()
