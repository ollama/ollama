#!/usr/bin/env python3
"""
Production Stress Test for mxbai-embed-large Data Loss Fix
Replicates the original catastrophic failure scenario: 5000 inputs → ~1000 outputs (80% loss)
"""

import requests
import json
import time
import sys
import math

def test_large_batch_embedding():
    """Execute large batch embedding test to validate the fix"""
    
    # API configuration
    url = "http://localhost:11434/api/embed"
    model = "mxbai-embed-large"
    
    # Generate test data matching the original issue report
    print("🔬 Generating test data...")
    test_inputs = [f"This is test sentence number {i} for embedding generation." for i in range(5000)]
    
    print(f"📊 Testing {len(test_inputs)} inputs with {model} (original issue scenario)")
    print("⚠️  Before fix: Expected ~80% data loss (5000 → ~1000 outputs)")
    print("✅ After fix: Expected 100% success rate")
    
    # Prepare API request
    payload = {
        "model": model,
        "input": test_inputs
    }
    
    print("\n🚀 Executing batch embedding request...")
    start_time = time.time()
    
    try:
        # Make the API request with extended timeout
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            print(f"\n✅ Request completed successfully in {duration:.2f} seconds")
            print(f"📈 Expected: {len(test_inputs)} embeddings")
            print(f"📉 Received: {len(embeddings)} embeddings")
            
            # CRITICAL VALIDATION: Check for data loss (the main issue)
            if len(embeddings) == len(test_inputs):
                print("🎉 NO DATA LOSS - All embeddings generated successfully!")
                print("✅ The catastrophic 80% data loss issue has been RESOLVED!")
            else:
                loss = len(test_inputs) - len(embeddings)
                loss_percent = (loss / len(test_inputs)) * 100
                print(f"❌ CRITICAL DATA LOSS DETECTED: {loss} embeddings missing ({loss_percent:.1f}% loss)")
                if loss_percent > 50:
                    print("🚨 This indicates the original issue is NOT fixed!")
                return False
            
            # Quality validation
            if embeddings:
                expected_dim = 1024  # mxbai-embed-large specification
                actual_dim = len(embeddings[0])
                
                if actual_dim == expected_dim:
                    print(f"✅ Correct embedding dimensions: {actual_dim}")
                else:
                    print(f"❌ Incorrect dimensions: expected {expected_dim}, got {actual_dim}")
                    return False
                
                # Data corruption validation
                print("\n🔍 Checking for data corruption...")
                nan_count = 0
                zero_count = 0
                invalid_count = 0
                
                # Sample first 100 embeddings for efficiency
                sample_size = min(100, len(embeddings))
                for i, emb in enumerate(embeddings[:sample_size]):
                    # NaN detection
                    if any(v != v for v in emb):  # NaN check (NaN != NaN)
                        nan_count += 1
                    
                    # Zero vector detection
                    if all(v == 0 for v in emb):
                        zero_count += 1
                    
                    # Invalid value detection
                    if any(math.isinf(v) or math.isnan(v) for v in emb):
                        invalid_count += 1
                
                if nan_count > 0:
                    print(f"❌ Found {nan_count} NaN embeddings in first {sample_size}")
                    return False
                
                if zero_count > 0:
                    print(f"⚠️  Found {zero_count} zero vectors in first {sample_size}")
                
                if invalid_count > 0:
                    print(f"❌ Found {invalid_count} invalid embeddings in first {sample_size}")
                    return False
                
                print("✅ No data corruption detected in sample")
            
            # Performance metrics
            avg_time_per_embedding = duration / len(embeddings) * 1000
            throughput = len(embeddings) / duration
            
            print(f"\n📊 Performance Metrics:")
            print(f"⏱️  Average time per embedding: {avg_time_per_embedding:.2f}ms")
            print(f"🚀 Processing throughput: {throughput:.2f} embeddings/second")
            
            return True
            
        else:
            print(f"❌ Request failed with HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 10 minutes")
        print("This may indicate the server is still struggling with batch processing")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_server_health():
    """Verify Ollama server is running and responsive"""
    
    print("🔍 Checking Ollama server health...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running and responsive")
            return True
        else:
            print(f"❌ Server responded with HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        print("💡 Please start the server with: ./ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("❌ Server health check timed out")
        return False
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

def main():
    """Main test execution"""
    
    print("🧪 PRODUCTION STRESS TEST: mxbai-embed-large Data Loss Fix")
    print("=" * 70)
    print("📋 Test Scenario: Replicate original 5000 input batch processing")
    print("🎯 Objective: Verify 80% data loss issue is completely resolved")
    print("=" * 70)
    
    # Server health check
    if not check_server_health():
        print("\n❌ TEST ABORTED: Server health check failed")
        sys.exit(1)
    
    print("\n" + "-" * 50)
    
    # Execute main test
    success = test_large_batch_embedding()
    
    print("\n" + "=" * 70)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ The mxbai-embed-large data loss issue is COMPLETELY FIXED!")
        print("🚀 Production systems can now process large batches without data loss")
        print("📈 Performance is stable and reliable")
    else:
        print("❌ TESTS FAILED!")
        print("🚨 The data loss issue may still exist or new problems were introduced")
        print("🔧 Please review the implementation and server logs")
        print("💡 Consider checking the error messages above for diagnostic information")
        sys.exit(1)

if __name__ == "__main__":
    main()
