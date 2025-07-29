#!/usr/bin/env python3
"""
Test script for the Syllable API microservice.
"""

import sys

import requests

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
                data = response.json()
                syllables = data["syllables"]
                status = "✅" if syllables == expected else "⚠️"
                print(f"   {status} '{text}': {syllables} syllables (expected {expected})")
            else:
                print(f"   ❌ '{text}': Error {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ '{text}': {e}")

def test_detailed_syllables():
    """Test the detailed syllables endpoint."""
    print("\n📊 Testing detailed syllables endpoint...")
    
    test_cases = [
        "hello world",
        "beautiful programming language",
        "artificial intelligence machine learning"
    ]
    
    for text in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/syllables",
                json={"text": text},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ '{text}':")
                print(f"      Total syllables: {data['syllables']}")
                for word_info in data['words']:
                    print(f"      - {word_info['word']}: {word_info['syllables']} syllables")
            else:
                print(f"   ❌ '{text}': Error {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ '{text}': {e}")

def test_haiku_syllables():
    """Test the haiku syllables endpoint."""
    print("\n🌸 Testing haiku syllables endpoint...")
    
    haiku = """Cherry blossoms fall
Softly on the peaceful earth
Spring has come again"""
    
    try:
        response = requests.post(
            f"{BASE_URL}/syllables/haiku",
            json={"text": haiku},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Haiku analysis:")
            for line_info in data['lines']:
                syllables = line_info['syllables']
                line = line_info['line']
                status = "✅" if syllables == 5 or syllables == 7 else "⚠️"
                print(f"      {status} '{line}': {syllables} syllables")
        else:
            print(f"   ❌ Haiku test: Error {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Haiku test: {e}")

def main():
    """Run all tests."""
    print("🧪 Syllable API Test Suite")
    print("=" * 50)
    
    # Test if service is running
    if not test_health():
        print("\n❌ Service is not running. Please start the API first:")
        print("   uv run python main.py")
        sys.exit(1)
    
    # Run all tests
    test_simple_syllables()
    test_detailed_syllables()
    test_haiku_syllables()
    
    print("\n🎉 Test suite completed!")

if __name__ == "__main__":
    main() 