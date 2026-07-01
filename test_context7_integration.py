#!/usr/bin/env python3
"""
Test script for Context7 integration.
Verifies the library docs context functions work correctly.
"""

import asyncio
import os
import sys

# Add api-gateway to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api-gateway'))

# Set test environment variables
os.environ['DOCS_CONTEXT_ENABLED'] = 'true'
os.environ['DOCS_CONTEXT_MAX_CHARS'] = '4000'
os.environ['REDIS_URL'] = 'redis://localhost:6379'  # Assumes Redis is running

async def test_library_detection():
    """Test library name detection from messages."""
    from main import _guess_library_from_message
    
    print("Testing library detection...")
    
    tests = [
        ("How do I use FastAPI streaming?", "fastapi"),
        ("What's new in React 19?", "react"),
        ("Next.js server actions tutorial", "next.js"),
        ("Configure Redis connection pool", "redis"),
        ("Install PyTorch on Windows", "pytorch"),
        ("General programming question", ""),
    ]
    
    for message, expected in tests:
        result = _guess_library_from_message(message)
        status = "✓" if result.lower() == expected.lower() else "✗"
        print(f"{status} '{message}' -> '{result}' (expected: '{expected}')")

async def test_should_fetch_docs():
    """Test heuristics for when to fetch docs."""
    from main import should_fetch_library_docs
    
    print("\nTesting fetch decision heuristics...")
    
    # Mock session without GitHub context
    session_no_github = {"messages": [], "model": "llama3.2:1b"}
    
    # Mock session with GitHub context
    session_with_github = {
        "messages": [],
        "model": "llama3.2:1b",
        "github_owner": "test",
        "github_repo": "test-repo"
    }
    
    tests = [
        ("How do I use Ollama streaming API?", session_no_github, True),
        ("use docs to explain FastAPI", session_no_github, True),
        ("hi", session_no_github, False),
        ("What is this repo about?", session_with_github, False),
        ("How does React work?", session_no_github, True),
    ]
    
    for message, session, expected in tests:
        result = should_fetch_library_docs(message, session)
        status = "✓" if result == expected else "✗"
        github = "with GitHub" if "github_owner" in session else "no GitHub"
        print(f"{status} '{message[:40]}' ({github}) -> {result} (expected: {expected})")

async def test_docs_fetch():
    """Test actual docs fetching (requires network)."""
    import redis.asyncio as redis
    from main import build_library_docs_context, redis_client, REDIS_URL
    
    print("\nTesting actual docs fetch (requires network and Redis)...")
    
    # Try to connect to Redis
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        print("✓ Redis connection successful")
        
        # Set redis_client globally (normally done at startup)
        import main
        main.redis_client = client
        
        # Test fetching docs
        message = "How do I use FastAPI streaming responses?"
        print(f"\nFetching docs for: '{message}'")
        
        result = await build_library_docs_context(message)
        
        if result:
            print(f"✓ Docs fetched successfully ({len(result)} chars)")
            print(f"\nFirst 200 chars of response:")
            print(result[:200] + "...")
        else:
            print("✗ No docs returned (check CONTEXT7_API_KEY or rate limits)")
        
        await client.aclose()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure Redis is running: docker compose up -d redis")

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Context7 Integration Test Suite")
    print("=" * 60)
    
    await test_library_detection()
    await test_should_fetch_docs()
    
    # Only run network test if explicitly requested
    if "--network" in sys.argv:
        await test_docs_fetch()
    else:
        print("\nSkipping network test (pass --network to enable)")
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
