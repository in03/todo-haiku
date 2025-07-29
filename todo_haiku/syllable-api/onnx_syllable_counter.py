#!/usr/bin/env python3
"""
ONNX-based syllable counter with dictionary fallback
"""

import json
import string
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
import onnxruntime as ort


class SyllableCounter(ABC):
    """Counts syllables in a given word."""

    @abstractmethod
    def count_syllables(self, word: str) -> Tuple[int, ...]:
        """Counts syllables in the given word. Returns a tuple of counts to
        account for multiple possible pronunciations. An empty tuple is returned
        if the counter is unable to count syllables.
        """
        pass

class CmudictSyllableCounter(SyllableCounter):
    """Counts syllables using the CMUdict pronunciation dictionary."""

    def __init__(self, file="cmudict/cmudict.dict"):
        self.d = self._read_from(file)

    @staticmethod
    def _read_from(file):
        d = {}
        try:
            with open(file) as f:
                for line in f:
                    word, *pieces = line.split()
                    if '(' in word:  # Alternate pronunciation
                        word = word[:word.index('(')]
                    syllables = sum(p[-1].isdigit() for p in pieces)
                    d.setdefault(word, []).append(syllables)
            for w, cnts in d.items():
                d[w] = tuple(sorted(set(cnts)))
        except FileNotFoundError:
            print("⚠️  CMU dictionary file not found, proceeding without dictionary lookup")
            d = {}
        return d

    def count_syllables(self, word: str) -> Tuple[int, ...]:
        word = word.lower().lstrip(string.punctuation)
        counts = self.d.get(word)
        if counts:
            return counts
        word = word.rstrip(string.punctuation)
        return self.d.get(word) or ()

class OnnxSyllableCounter(SyllableCounter):
    """Counts syllables using an ONNX model - Fast ML inference."""

    def __init__(self, model_path="syllable_model.onnx", metadata_path="model_metadata.json"):
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            self.chars = metadata["character_encoding"]["alphabet"]
            self.max_len = metadata["character_encoding"]["max_word_length"]
            self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
            self.trimchars = ''.join(set(string.punctuation) - set(self.chars))
            
            print("✅ Successfully loaded ONNX syllable model")
            self.model_available = True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            self.model_available = False

    def encode_word(self, word):
        """Encode a word to one-hot tensor."""
        word = word.lower()[:self.max_len]  # Lowercase and truncate
        
        # Create zero tensor
        encoded = np.zeros((1, self.max_len, len(self.chars)), dtype=np.float32)
        
        # Set one-hot values
        for i, char in enumerate(word):
            if char in self.char_to_idx:
                encoded[0, i, self.char_to_idx[char]] = 1.0
        
        return encoded

    def count_syllables(self, word: str) -> Tuple[int, ...]:
        if not self.model_available:
            return ()
            
        word = word.lower().strip(self.trimchars)
        if count := self._count(word):
            return (int(round(count)),)
        return ()

    def _count(self, word):
        if not word or len(word) > self.max_len:
            return None
        
        # Check if all characters are in our vocabulary
        if not all(c in self.chars for c in word):
            return None
        
        try:
            encoded = self.encode_word(word)
            
            # Run inference
            result = self.session.run([self.output_name], {self.input_name: encoded})
            
            # Round to nearest integer, minimum 1
            syllables = max(1, float(result[0][0][0]))
            return syllables
            
        except Exception as e:
            print(f"ONNX model prediction failed: {e}")
            return None

class CompositeSyllableCounter(SyllableCounter):
    """Counts syllables delegating to other syllable counters. Each delegate is
    tried in order, the first non-empty response is returned.
    """
    def __init__(self, delegates: Iterable[SyllableCounter]):
        self.delegates = delegates

    def count_syllables(self, word: str) -> Tuple[int, ...]:
        for delegate in self.delegates:
            if counts := delegate.count_syllables(word):
                return counts
        return ()

# Main syllable service class
class SyllableService:
    """Syllable counting service using ONNX model with CMU dictionary fallback."""
    
    def __init__(self):
        # Initialize counters: dictionary first, then ONNX model
        cmu_counter = CmudictSyllableCounter()
        onnx_counter = OnnxSyllableCounter()
        self.counter = CompositeSyllableCounter([cmu_counter, onnx_counter])
    
    def count_syllables(self, text: str) -> int:
        """Count syllables in text, returning total count."""
        if not text or not text.strip():
            return 0
        
        # Split into words and count each
        words = text.split()
        total = 0
        
        for word in words:
            # Remove punctuation for counting
            clean_word = word.strip(string.punctuation)
            if clean_word:
                syllables = self.counter.count_syllables(clean_word)
                total += syllables[0] if syllables else 1  # Default to 1 if no count available
        
        return total

# Test the implementation
def test_syllable_library():
    """Test the ONNX-based syllable library implementation."""
    print("🚀 Testing ONNX Syllable Library (CMU Dictionary + ONNX Model)")
    print("=" * 70)
    
    try:
        # Initialize the syllable service
        service = SyllableService()
        
        test_words = [
            "hello", "world", "beautiful", "sophisticated", "cat",
            "computer", "extraordinary", "spring", "cherry", "blossom",
            "mountain", "peaceful", "family", "novelword",
            "supercalifragilisticexpialidocious", "antidisestablishmentarianism"
        ]
        
        print("\n📊 Testing composite counter (CMU Dictionary + ONNX Model):")
        print("-" * 70)
        
        for word in test_words:
            try:
                syllables = service.count_syllables(word)
                print(f"{word:30} | {syllables:2} syllables")
            except Exception as e:
                print(f"{word:30} | ERROR: {e}")
        
        return service
        
    except Exception as e:
        print(f"❌ Failed to initialize syllable library: {e}")
        return None

if __name__ == "__main__":
    test_syllable_library() 