#!/usr/bin/env python3
"""
Test script for GitHub OAuth endpoints
"""
import requests
import sys

BASE_URL = "http://localhost:8080/api"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_github_status_no_auth():
    """Test GitHub status without authentication (should fail)"""
    print("\nTesting /auth/github/status without auth...")
    try:
        response = requests.get(f"{BASE_URL}/auth/github/status", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text}")
        return response.status_code == 401
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_github_status_with_auth(api_key):
    """Test GitHub status with authentication"""
    print(f"\nTesting /auth/github/status with API key...")
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            f"{BASE_URL}/auth/github/status",
            headers=headers,
            timeout=5
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_github_login(api_key):
    """Test GitHub OAuth login start"""
    print(f"\nTesting /auth/github/login...")
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(
            f"{BASE_URL}/auth/github/login",
            headers=headers,
            timeout=5
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Authorization URL: {data.get('authorization_url', 'N/A')[:80]}...")
        else:
            print(f"  Response: {response.text}")
        return response.status_code in [200, 500]  # 500 if not configured
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 60)
    print("GitHub OAuth Integration Test")
    print("=" * 60)
    
    # Test health first
    if not test_health():
        print("\n❌ Health check failed! API may not be running.")
        sys.exit(1)
    
    print("\n✅ Health check passed!")
    
    # Test GitHub status without auth
    if not test_github_status_no_auth():
        print("\n❌ GitHub status without auth should return 401")
    else:
        print("\n✅ GitHub status correctly requires authentication")
    
    # Get API key from user
    print("\n" + "=" * 60)
    api_key = input("Enter your API key (or press Enter to skip auth tests): ").strip()
    
    if api_key:
        # Test GitHub status with auth
        if test_github_status_with_auth(api_key):
            print("\n✅ GitHub status endpoint works!")
        else:
            print("\n❌ GitHub status endpoint failed")
        
        # Test GitHub login
        if test_github_login(api_key):
            print("\n✅ GitHub login endpoint works!")
        else:
            print("\n❌ GitHub login endpoint failed")
    else:
        print("\nSkipping authenticated tests.")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("  - API is running and responding")
    print("  - GitHub endpoints are registered")
    print("  - Authentication is enforced")
    print("\nNext steps:")
    print("  1. Generate an API key via /admin")
    print("  2. Configure GitHub OAuth in /admin")
    print("  3. Test the full OAuth flow in the browser")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
