# Chat Storage Fix - Stable User IDs and Improved Persistence

## Problem Summary

Chats were not being stored properly for API keys due to:

1. **Unstable user IDs**: Used `hash(api_key)` which changes on process restart (Python hash randomization)
2. **Streaming save issues**: Line buffering problems and incomplete saves
3. **Frontend sync**: UI didn't reload from server after streaming

## Changes Made

### 1. Stable User ID Mapping (`api-gateway/main.py`)

**Before:**
```python
return {key: f"user_{abs(hash(key)) % 10000:04d}" for key in keys}
```
- Hash-based user_id changed every restart
- Same API key → different user_id → can't find old sessions

**After:**
```python
async def get_or_create_user_id_for_key(api_key: str) -> str:
    # Check Redis for existing mapping
    user_id = await redis_client.hget(API_KEY_USER_MAP_KEY, api_key)
    if user_id:
        return user_id
    
    # Create stable UUID-based user_id
    new_user_id = f"user_{uuid.uuid4().hex[:12]}"
    await redis_client.hset(API_KEY_USER_MAP_KEY, api_key, new_user_id)
    return new_user_id
```

**Benefits:**
- Stable user_id across restarts
- One-to-one mapping stored in Redis: `gateway:api_key_user_map`
- Existing sessions remain accessible after container restart

### 2. Improved Streaming Save (`api-gateway/main.py`)

**Fixed:**
- Added proper line buffering (prevents mid-line splits breaking JSON parsing)
- Always save session, even if response is empty
- Forward chunks immediately while accumulating in parallel

**Before:**
```python
async for chunk in response.aiter_bytes():
    chunk_str = chunk.decode("utf-8")
    for line in chunk_str.split("\n"):  # Can split mid-line!
        if line.startswith("data: "):
            # ...parse
    yield chunk
```

**After:**
```python
line_buffer = ""
async for chunk in response.aiter_bytes():
    yield chunk  # Forward immediately
    
    line_buffer += chunk.decode("utf-8")
    while "\n" in line_buffer:
        line, line_buffer = line_buffer.split("\n", 1)  # Proper buffering
        if line.startswith("data: "):
            # ...parse reliably
```

**Always saves:**
```python
finally:
    await client.aclose()
    # Always save (user message already in array)
    if accumulated_content:
        messages.append({"role": "assistant", "content": accumulated_content})
    await update_session(user_id, session_id, messages, model)
```

### 3. Frontend Reload After Streaming (`Chat.tsx`)

**Changed:**
```typescript
// After streaming completes
setStreamingMessage('');

// Reload from server to ensure sync with Redis
await loadMessages();
```

**Benefits:**
- UI matches Redis state after every message
- Catches any server-side save issues immediately
- Ensures consistency on page refresh

### 4. Cleanup on Key Deletion

When revoking API keys via admin:
```python
# Remove from user mapping
removed_key = keys[index]
await redis_client.hdel(API_KEY_USER_MAP_KEY, removed_key)
```

## Redis Keys

### New Key:
- `gateway:api_key_user_map` (hash) - Maps API keys to stable user IDs

### Existing Keys (unchanged):
- `session:{user_id}:{session_id}` - Session data
- `sessions:{user_id}` - Set of session IDs per user

## Migration Path

### Automatic Migration
Keys are migrated automatically on first use:
1. Existing hash-based user_ids still work in Redis
2. On next API request, stable mapping is created
3. New sessions use stable user_id
4. Old sessions remain under old user_id (orphaned but accessible via direct key)

### Manual Migration (Optional)
If you need to consolidate old sessions:

```bash
# 1. Find old sessions
docker compose exec redis redis-cli KEYS "session:user_*"

# 2. For each API key, map old hash-based ID to new stable ID
# (Would require custom migration script)
```

## Testing

### Verify Stable User IDs

1. **Create session with API key:**
   ```bash
   curl -X POST http://localhost:8080/api/sessions \
     -H "Authorization: Bearer YOUR_KEY" \
     -H "Content-Type: application/json" \
     -d '{"title": "Test", "model": "llama3.2:1b"}'
   ```

2. **Check user mapping:**
   ```bash
   docker compose exec redis redis-cli HGETALL gateway:api_key_user_map
   ```

3. **Restart container:**
   ```bash
   docker compose restart api
   ```

4. **List sessions again (should see same session):**
   ```bash
   curl -X GET http://localhost:8080/api/sessions \
     -H "Authorization: Bearer YOUR_KEY"
   ```

### Verify Streaming Saves

1. Send a message via UI (uses streaming)
2. Refresh the page
3. Message should still be visible

### Verify Redis Persistence

```bash
# Check sessions exist
docker compose exec redis redis-cli KEYS "session:*"

# Check specific session messages
docker compose exec redis redis-cli HGET "session:user_XXXXX:SESSION_ID" messages
```

## Backward Compatibility

- ✅ Old API keys continue to work
- ✅ Non-streaming mode unchanged
- ✅ `.env` API keys work (migrated on first use)
- ✅ Admin-generated keys work immediately
- ⚠️ Old sessions remain under old hash-based user_ids (orphaned but still in Redis)

## Performance Impact

- **Minimal**: One additional Redis HGET per auth (cached by `configured_api_keys()`)
- **Storage**: ~50 bytes per API key in mapping hash
- **Streaming**: Slightly more memory for line buffer (negligible)

## Rollback

If issues arise, revert to hash-based (not recommended):

```python
async def configured_api_keys() -> dict[str, str]:
    raw_keys = await get_config_value("API_KEYS", os.getenv("API_KEYS", ""))
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    return {key: f"user_{abs(hash(key)) % 10000:04d}" for key in keys}
```

Sessions will switch back to hash-based user_ids (breaking stability again).

## Future Improvements

1. **Migration tool** to consolidate orphaned sessions under new user_ids
2. **User management** UI to see all users and their sessions
3. **Session transfer** between user_ids
4. **Analytics** per user_id
