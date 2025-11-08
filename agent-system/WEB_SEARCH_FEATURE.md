# Web Search Feature

## Overview
Your Agent Terminal System now has internet access via DuckDuckGo web search!

## How It Works

### For Agents
Agents automatically know they can search the internet. When they need current information, they include a search tag in their response:

```
[SEARCH: your query here]
```

The system will:
1. Detect the search tag
2. Execute the search via DuckDuckGo
3. Replace the tag with formatted results
4. Return the enhanced response with search results

### For You
You can also use the search API directly:

```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query"}'
```

## Example Agent Interaction

**You**: "What's happening with AI in 2025?"

**Researcher Agent**: "Let me check that for you. [SEARCH: AI developments 2025]"

**Response**:
```
Let me check that for you.

🔍 SEARCH RESULTS for "AI developments 2025":

1. Latest AI Breakthroughs in 2025
   Major advancements in artificial intelligence including...
   Source: https://example.com/ai-news

2. AI Industry Trends 2025
   The AI industry continues to grow with new models...
   Source: https://example.com/trends

[... up to 5 results ...]
```

## Features

- **Automatic Detection**: Agents can request searches on their own
- **Rate Limiting**: Built-in 2-second cooldown between searches
- **Top 5 Results**: Returns the most relevant results
- **Clean Formatting**: Results are formatted with title, description, and URL

## Rate Limiting Note

DuckDuckGo may rate limit requests from certain IPs (like GitHub Codespaces). If you see:
```
⚠️ Search rate limited - try again in a few seconds
```

This is normal. The search will work when used sparingly or from a local machine.

## API Endpoints

### POST /api/search
Search the web directly

**Request:**
```json
{
  "query": "your search query"
}
```

**Response:**
```json
{
  "query": "your search query",
  "count": 5,
  "results": [
    {
      "title": "Result Title",
      "description": "Result description...",
      "url": "https://example.com"
    }
  ]
}
```

### POST /api/message
Send a message to an agent (searches are processed automatically)

**Request:**
```json
{
  "agent": "researcher",
  "message": "What's the weather like today?",
  "model": "qwen2.5:0.5b"
}
```

If the agent responds with `[SEARCH: weather today]`, the search results will be automatically included in the response.

## Technical Details

- **Library**: duck-duck-scrape (no API key needed)
- **Rate Limit**: 2 seconds between searches
- **Max Results**: 5 per search
- **Timeout**: 30 seconds per search
- **Locale**: en-us
- **Safe Search**: Disabled (set to 0)

## Tips

1. Agents work best when they know they can search - ask open-ended questions
2. The Researcher agent is especially good at using web search
3. Searches are free and require no API keys
4. If rate limited, wait a minute before trying again
5. Works best on local machines or non-shared IPs

## Future Enhancements

Potential improvements in future phases:
- Alternative search providers (SerpAPI, Brave Search)
- Search result caching
- Multi-query search
- Image search support
- News-specific search
