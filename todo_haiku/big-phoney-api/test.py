#!/usr/bin/env python3
"""
Test script for the Big Phoney API microservice.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"  # Change to your deployed URL when testing

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_simple_syllables():
    """Test the simple syllables endpoint."""
    print("\n📝 Testing simple syllables endpoint...")
    
    test_cases = [
        ("hello", 2),
        ("world", 1),
        ("beautiful", 3),
        ("sophisticated", 5),
        ("cat", 1),
        ("computer", 3),
        ("extraordinary", 6),
        ("hello world", 3),
        ("beautiful day", 4)
    ]
    
    for text, expected in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/syllables/simple",
                json={"text": text},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                syllables = result.get("syllables")
                if syllables == expected:
                    print(f"✅ '{text}' -> {syllables} syllables (expected {expected})")
                else:
                    print(f"⚠️  '{text}' -> {syllables} syllables (expected {expected})")
            else:
                print(f"❌ '{text}' failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ '{text}' error: {e}")

def test_detailed_syllables():
    """Test the detailed syllables endpoint."""
    print("\n🔍 Testing detailed syllables endpoint...")
    
    test_text = "hello beautiful world"
    
    try:
        response = requests.post(
            f"{BASE_URL}/syllables",
            json={"text": test_text},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Text: '{test_text}'")
            print(f"   Total syllables: {result.get('syllables')}")
            print(f"   Word breakdown:")
            for word_info in result.get("words", []):
                print(f"     - '{word_info['word']}': {word_info['syllables']} syllables")
        else:
            print(f"❌ Detailed syllables failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Detailed syllables error: {e}")

def test_error_handling():
    """Test error handling."""
    print("\n🚨 Testing error handling...")
    
    # Test empty text
    try:
        response = requests.post(
            f"{BASE_URL}/syllables/simple",
            json={"text": ""},
            timeout=5
        )
        
        if response.status_code == 400:
            print("✅ Empty text correctly rejected")
        else:
            print(f"⚠️  Empty text returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Empty text test error: {e}")
    
    # Test missing text field
    try:
        response = requests.post(
            f"{BASE_URL}/syllables/simple",
            json={},
            timeout=5
        )
        
        if response.status_code == 422:  # Validation error
            print("✅ Missing text field correctly rejected")
        else:
            print(f"⚠️  Missing text field returned status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Missing text field test error: {e}")

def main():
    """Run all tests."""
    print("🧪 Big Phoney API Test Suite")
    print("=" * 40)
    
    # Test health endpoint
    if not test_health():
        print("\n❌ Health check failed. Make sure the service is running.")
        sys.exit(1)
    
    # Test simple syllables
    test_simple_syllables()
    
    # Test detailed syllables
    test_detailed_syllables()
    
    # Test error handling
    test_error_handling()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main() 