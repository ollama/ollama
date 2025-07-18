#!/usr/bin/env python3

import requests
import json

# Test the cleaned reranking API
url = "http://localhost:11434/api/rerank"

# Simple test data using the community-validated format
payload = {
    "query": "What is machine learning?",
    "documents": [
        "Angela Merkel was the Chancellor of Germany",
        "Machine learning is a subset of artificial intelligence",
        "Pizza is made with tomatoes and cheese",
        "Deep learning uses neural networks for pattern recognition",
        "The weather today is sunny and warm"
    ],
    "top_n": 5,
    "model": "test-reranker:latest"
}

print("Testing cleaned reranking API...")
print(f"URL: {url}")
print(f"Payload: {json.dumps(payload, indent=2)}")

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Response structure:")
        print(json.dumps(result, indent=2))
        
        print(f"\nRanked Results:")
        for i, item in enumerate(result.get('results', [])):
            print(f"{i+1}. Score: {item['relevance_score']:.3f} - {item['document']}")
            
    else:
        print(f"Error Response: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON response: {e}")
    print(f"Raw response: {response.text}")
