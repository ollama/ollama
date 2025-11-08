# Web Search Setup Guide

Your agents can search the web for real-time information! This guide shows you how to enable Tavily AI search.

## Quick Setup (2 minutes)

### Step 1: Get Your Free Tavily API Key

1. Go to **https://tavily.com**
2. Click **"Get Started"** or **"Sign Up"**
3. Sign up with email or GitHub
4. Go to your **Dashboard**
5. Copy your **API Key**

**Free Tier:**
- 1,000 searches per month
- No credit card required
- Perfect for testing and development

### Step 2: Add API Key to Your Environment

Open the `.env` file in the `agent-system` directory:

```bash
cd agent-system
nano .env
```

Add your API key:

```env
TAVILY_API_KEY=tvly-your-actual-api-key-here
```

Save and exit (Ctrl+O, Enter, Ctrl+X in nano).

### Step 3: Restart the Server

```bash
npm start
```

You should see:
```
✓ Tavily AI search enabled
```

### Step 4: Test It!

Ask an agent to search for something:

```
Agent: Researcher
Message: "Search for the latest AI developments in 2025"
```

Or via API:
```bash
curl -X POST http://localhost:3000/api/message \
  -H "Content-Type: application/json" \
  -d '{"agent":"researcher","message":"What are the latest React 19 features?"}'
```

The agent will automatically search and include real-time results!

---

## How It Works

### Automatic Search Detection

Agents are instructed to use this syntax when they need current information:

```
[SEARCH: your search query]
```

The system automatically:
1. Detects `[SEARCH: ...]` tags in responses
2. Performs the search via Tavily API
3. Replaces the tag with actual search results
4. Returns the enhanced response to you

### Example Agent Response

**Before processing:**
```
Let me check that for you. [SEARCH: latest TypeScript features 2025]
```

**After processing:**
```
Let me check that for you.

🔍 SEARCH RESULTS for "latest TypeScript features 2025":

1. TypeScript 5.3 Release Notes
   New features include import attributes, resolution mode...
   Source: https://devblogs.microsoft.com/typescript/

2. What's New in TypeScript - 2025 Update
   The latest TypeScript releases bring enhanced type inference...
   Source: https://typescriptlang.org/docs/
```

---

## Rate Limiting

To prevent abuse and stay within API limits:

- **Default:** 10 searches per minute
- **Configurable** in `.env`:

```env
SEARCH_RATE_LIMIT=10  # Adjust as needed
```

When rate limit is reached, the system waits automatically before continuing.

---

## Fallback to DuckDuckGo

**Without Tavily API key:**
- System falls back to DuckDuckGo scraping
- Subject to rate limiting and captchas
- Less reliable

**With Tavily API key:**
- Fast, reliable results
- AI-optimized content
- No captchas or blocks
- Better quality

---

## Configuration Options

Edit `.env` to customize:

```env
# API Key (required for best results)
TAVILY_API_KEY=your_api_key

# Rate limiting (searches per minute)
SEARCH_RATE_LIMIT=10

# Max results per search
SEARCH_MAX_RESULTS=5

# Search timeout (milliseconds)
SEARCH_TIMEOUT=10000
```

---

## Troubleshooting

### "No Tavily API key found"

**Solution:** Add your API key to `.env` and restart the server.

### "Rate limit reached"

**Solution:** Wait 60 seconds or increase `SEARCH_RATE_LIMIT` in `.env`.

### "Tavily failed, falling back to DuckDuckGo"

**Possible causes:**
- Invalid API key
- Network issues
- API timeout

**Solution:**
1. Verify API key is correct
2. Check internet connection
3. Check Tavily status at https://status.tavily.com

### "DuckDuckGo detected an anomaly"

**Solution:** This means DuckDuckGo is blocking scraping. Add a Tavily API key to fix this.

---

## Testing Search Directly

Test the search API endpoint:

```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"artificial intelligence news"}'
```

Expected response:
```json
{
  "query": "artificial intelligence news",
  "results": [
    {
      "title": "Latest AI News...",
      "description": "Recent developments in...",
      "url": "https://...",
      "score": 0.95
    }
  ],
  "count": 5
}
```

---

## Best Practices

1. **Be Specific:** Better searches get better results
   - ❌ "AI"
   - ✅ "latest GPT-4 features December 2025"

2. **Ask Agents to Search:** Don't do it manually
   - Let agents decide when they need current info
   - They'll automatically include searches when needed

3. **Monitor Usage:** Free tier is 1,000/month
   - Each agent search = 1 API call
   - Collaboration can use multiple searches

4. **Use DuckDuckGo for Development:** If you're just testing
   - No API key needed
   - Good for initial development
   - Add Tavily when you need reliability

---

## Alternative: Brave Search API

If you prefer Brave Search:

1. Get API key at https://brave.com/search/api/
2. Free tier: 2,000 queries/month
3. Update `server.js` to use Brave instead

---

## Need Help?

- Tavily Docs: https://docs.tavily.com
- Tavily Support: support@tavily.com
- Check server logs: `tail -f /tmp/server.log`

**Happy searching!** 🔍
