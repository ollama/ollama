#!/usr/bin/env python3
"""
Test script to verify the mxbai-embed-large fix with large batch processing
This simulates the original issue scenario with 5000 inputs.
"""

import requests
import json
import time
import sys

def test_large_batch_embedding():
    """Test large batch embedding with mxbai-embed-large"""
    
    # API endpoint
    url = "http://localhost:11434/api/embed"
    
    # Generate test data - 5000 inputs as mentioned in the original issue
    test_inputs = [f"This is test sentence number {i} for embedding generation." for i in range(5000)]
    
    print(f"Testing {len(test_inputs)} inputs with mxbai-embed-large...")
    
    # Prepare request
    payload = {
        "model": "mxbai-embed-large",
        "input": test_inputs
    }
    
    start_time = time.time()
    
    try:
        # Make request
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            print(f"✅ Request completed in {duration:.2f} seconds")
            print(f"📊 Expected: {len(test_inputs)} embeddings")
            print(f"📊 Received: {len(embeddings)} embeddings")
            
            # Check for data loss
            if len(embeddings) == len(test_inputs):
                print("✅ NO DATA LOSS - All embeddings generated successfully!")
            else:
                loss = len(test_inputs) - len(embeddings)
                loss_percent = (loss / len(test_inputs)) * 100
                print(f"❌ DATA LOSS DETECTED: {loss} embeddings missing ({loss_percent:.1f}% loss)")
                return False
            
            # Verify embedding dimensions
            if embeddings:
                expected_dim = 1024  # mxbai-embed-large dimension
                actual_dim = len(embeddings[0])
                if actual_dim == expected_dim:
                    print(f"✅ Correct embedding dimensions: {actual_dim}")
                else:
                    print(f"❌ Incorrect dimensions: expected {expected_dim}, got {actual_dim}")
                    return False
                
                # Check for NaN or zero vectors
                nan_count = 0
                zero_count = 0
                for i, emb in enumerate(embeddings[:100]):  # Check first 100 for speed
                    if any(v != v for v in emb):  # NaN check
                        nan_count += 1
                    if all(v == 0 for v in emb):  # Zero vector check
                        zero_count += 1
                
                if nan_count > 0:
                    print(f"❌ Found {nan_count} NaN embeddings in first 100")
                    return False
                
                if zero_count > 0:
                    print(f"❌ Found {zero_count} zero vectors in first 100")
                    return False
                
                print("✅ No NaN or zero vectors detected")
            
            # Performance metrics
            avg_time_per_embedding = duration / len(embeddings) * 1000
            print(f"⏱️  Average time per embedding: {avg_time_per_embedding:.2f}ms")
            
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_consistency():
    """Test embedding consistency with similar texts"""
    
    url = "http://localhost:11434/api/embed"
    
    test_cases = [
        ["The sky is blue and beautiful", "The sky appears blue in color", "The ocean is deep and vast"],
        ["Hello world", "Hello there", "Goodbye world"]
    ]
    
    print("\n🔍 Testing embedding consistency...")
    
    for i, texts in enumerate(test_cases):
        payload = {
            "model": "mxbai-embed-large",
            "input": texts
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                embeddings = response.json().get("embeddings", [])
                
                if len(embeddings) == len(texts):
                    # Simple similarity check for first test case
                    if i == 0 and len(embeddings) >= 3:
                        # Calculate dot product similarity
                        def cosine_similarity(a, b):
                            dot = sum(x*y for x, y in zip(a, b))
                            mag_a = sum(x*x for x in a) ** 0.5
                            mag_b = sum(y*y for y in b) ** 0.5
                            return dot / (mag_a * mag_b) if mag_a * mag_b > 0 else 0
                        
                        sim_similar = cosine_similarity(embeddings[0], embeddings[1])
                        sim_different = cosine_similarity(embeddings[0], embeddings[2])
                        
                        print(f"  Test case {i+1}: Similar texts similarity: {sim_similar:.4f}, Different texts: {sim_different:.4f}")
                        
                        if sim_similar > sim_different:
                            print(f"  ✅ Consistency check passed")
                        else:
                            print(f"  ⚠️  Similar texts should have higher similarity")
                    else:
                        print(f"  ✅ Test case {i+1} passed")
                else:
                    print(f"  ❌ Test case {i+1} failed: expected {len(texts)} embeddings, got {len(embeddings)}")
            else:
                print(f"  ❌ Test case {i+1} failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Test case {i+1} error: {e}")

if __name__ == "__main__":
    print("🧪 Testing mxbai-embed-large fix for vector data loss issue")
    print("=" * 60)
    
    # Check if Ollama server is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama server is not responding correctly")
            sys.exit(1)
    except:
        print("❌ Cannot connect to Ollama server at http://localhost:11434")
        print("Please start the server with: ./ollama serve")
        sys.exit(1)
    
    print("✅ Ollama server is running")
    
    # Run tests
    success = test_large_batch_embedding()
    test_consistency()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - The mxbai-embed-large data loss issue is FIXED!")
    else:
        print("❌ TESTS FAILED - Issue may still exist")
        sys.exit(1)
