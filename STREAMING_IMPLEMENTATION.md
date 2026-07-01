# Streaming Implementation - Performance Improvements

## Summary
Implemented streaming responses and GitHub context caching to reduce CPU usage from 400% and improve response latency. The UI now displays responses token-by-token instead of showing "Thinking..." until completion.

## Changes Made

### 1. Backend: GitHub Context Caching (`api-gateway/main.py`)

**Added:**
- `GITHUB_CONTEXT_CACHE_PREFIX` and `GITHUB_CONTEXT_CACHE_TTL` constants
- `_github_context_cache_key()` - generates cache keys for GitHub contexts
- `_invalidate_github_context_cache()` - invalidates cache when context changes

**Modified:**
- `build_github_context_for_chat()` - now checks Redis cache before fetching from GitHub API
  - Caches built context for 1 hour (configurable via `GITHUB_CONTEXT_CACHE_TTL`)
  - Reduces repeated GitHub API calls and prompt assembly overhead
  
**Cache Invalidation:**
- `set_session_github_context_endpoint()` - invalidates old cache when repo changes
- `add_session_github_files_endpoint()` - invalidates cache when files are added
- `clear_session_github_context_endpoint()` - invalidates cache when context is cleared

**Benefits:**
- Eliminates repeated GitHub API calls (tree fetch + up to 25 files)
- Reduces prompt assembly time on each message
- Significantly reduces CPU load for repo-linked chats

### 2. Backend: Streaming Support (`api-gateway/main.py`)

**Modified `session_chat()` endpoint:**
- Now checks `body.stream` flag
- **Streaming mode (`stream: true`):**
  - Uses `httpx.AsyncClient.stream()` for SSE forwarding
  - Parses SSE chunks to accumulate assistant content
  - Returns `StreamingResponse` with `media_type="text/event-stream"`
  - Saves accumulated message to session after streaming completes
- **Non-streaming mode (`stream: false`):**
  - Maintains original blocking behavior
  - Ensures backward compatibility

**Benefits:**
- Users see response tokens as they're generated
- Perceived latency drops significantly
- Time-to-first-token is now visible instead of hidden behind "Thinking..."

### 3. Frontend: Streaming API Client (`frontend/src/api.ts`)

**Added:**
- `sendMessageStreaming()` - async generator method
  - Accepts same parameters as `sendMessage()`
  - Yields content chunks as they arrive
  - Parses SSE data lines and extracts `delta.content`
  - Handles errors and cleanup properly

**Kept:**
- `sendMessage()` - original non-streaming method (unchanged for compatibility)

### 4. Frontend: UI Updates (`frontend/src/components/Chat.tsx`)

**Added:**
- `streamingMessage` state - tracks currently streaming assistant message

**Modified:**
- `handleSend()`:
  - Adds user message to local state immediately
  - Uses `sendMessageStreaming()` via async iterator
  - Updates `streamingMessage` state with each chunk
  - Adds complete assistant message to `messages` array after streaming
  - Handles errors by removing user message on failure

- Messages display:
  - Shows streaming message in real-time (if active)
  - Displays "Loading..." only when waiting for first token
  - Removed "Thinking..." static placeholder

**Benefits:**
- Users see responses appear token-by-token
- Better UX with immediate feedback
- Clear indication of model activity

## Behavior Preservation

All changes maintain backward compatibility:

1. **Non-streaming still works** - `stream: false` uses original code path
2. **Cache is transparent** - users see no difference except improved speed
3. **Error handling preserved** - all original error cases still handled
4. **Session management unchanged** - messages still saved to Redis
5. **GitHub integration works same way** - just faster with caching

## Performance Improvements

### Before:
- Every message rebuilt full GitHub context (API calls + file fetching)
- UI showed "Thinking..." for entire generation time
- CPU at 400% during prompt eval + generation
- High latency for repo-linked chats

### After:
- GitHub context cached (1 hour TTL, invalidated on changes)
- UI shows tokens as generated (streaming)
- CPU still high during inference, but work is reduced:
  - No repeated GitHub API calls
  - No repeated prompt assembly
  - Cache hits are instantaneous
- Much lower perceived latency

## Configuration

New environment variables in `api-gateway/.env`:

```bash
# GitHub context cache TTL (seconds)
GITHUB_CONTEXT_CACHE_TTL=3600  # 1 hour default
```

## Testing Recommendations

1. **Test streaming:**
   - Send a message and verify tokens appear incrementally
   - Check that full message is saved to session after completion

2. **Test caching:**
   - Link a GitHub repo and send a message (cache miss - slow)
   - Send another message (cache hit - fast)
   - Add files to context (cache invalidated)
   - Send message (cache miss again)

3. **Test backward compatibility:**
   - Ensure non-streaming mode still works if needed
   - Verify error handling works in both modes

4. **Monitor Redis:**
   - Check cache keys are created: `github:context:*`
   - Verify TTL is set (1 hour)
   - Confirm invalidation on context changes

## Next Steps (Optional Optimizations)

If CPU usage is still too high:

1. **Reduce context size for 1B model:**
   - Lower `MAX_REPO_CONTEXT_CHARS` (currently 16000)
   - Lower `MAX_REPO_FILES` (currently 25)
   
2. **Trim chat history:**
   - Keep only last N messages in session
   - Smaller prompts = faster eval

3. **Hardware optimization:**
   - Use GPU if available (`docker-compose.gpu.yml`)
   - Tune `OLLAMA_NUM_THREADS` to match CPU cores

4. **Model tuning:**
   - Set `num_ctx` to match actual usage
   - Cap `num_predict` for max tokens
