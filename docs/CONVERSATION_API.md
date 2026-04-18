"""
Conversation History API Documentation

This module provides comprehensive conversation management and retrieval endpoints.
Supports full conversation lifecycle: create, retrieve, update, delete, and search.

BASE PATH: /api/v1/conversations/

AUTHENTICATION:
All endpoints require user_id query parameter for authorization.
"""

# ============================================================================
# CONVERSATION MANAGEMENT ENDPOINTS
# ============================================================================

# GET /api/v1/conversations/
"""
List all conversations for a user.

QUERY PARAMETERS:
  - user_id (UUID, required): User ID to list conversations for
  - archived (bool, optional): Include archived conversations (default: false)
  - page (int, optional): Page number (1-indexed, default: 1)
  - page_size (int, optional): Items per page (1-100, default: 10)

RESPONSE:
  {
    "total": 25,
    "page": 1,
    "page_size": 10,
    "conversations": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "title": "Python debugging help",
        "model": "deepseek-coder:6.7b",
        "is_archived": false,
        "created_at": "2024-01-15T10:30:00Z",
        "accessed_at": "2024-01-15T14:22:15Z",
        "message_count": 12
      },
      ...
    ]
  }

EXAMPLE:
  GET /api/v1/conversations/?user_id=550e8400-e29b-41d4-a716-446655440000&page=1&page_size=20
"""

# POST /api/v1/conversations/
"""
Create a new conversation.

QUERY PARAMETERS:
  - user_id (UUID, required): User ID
  - model (str, required): Model name to use (e.g., "deepseek-coder:6.7b")
  - title (str, optional): Conversation title
  - system_prompt (str, optional): System instructions/context

RESPONSE:
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Code review session",
    "model": "deepseek-coder:6.7b",
    "system_prompt": "You are an expert code reviewer...",
    "is_archived": false,
    "created_at": "2024-01-15T10:30:00Z",
    "accessed_at": "2024-01-15T10:30:00Z"
  }

EXAMPLE:
  POST /api/v1/conversations/?user_id=550e8400-e29b-41d4-a716-446655440000&model=deepseek-coder:6.7b&title=Code%20review
"""

# GET /api/v1/conversations/{conversation_id}
"""
Get a specific conversation by ID.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)

RESPONSE:
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Code review session",
    "model": "deepseek-coder:6.7b",
    "system_prompt": "You are an expert code reviewer...",
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9
    },
    "is_archived": false,
    "created_at": "2024-01-15T10:30:00Z",
    "accessed_at": "2024-01-15T14:22:15Z",
    "message_count": 12
  }

NOTE: Automatically updates accessed_at timestamp

EXAMPLE:
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000?user_id=550e8400-e29b-41d4-a716-446655440001
"""

# PUT /api/v1/conversations/{conversation_id}
"""
Update conversation details.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)
  - title (str, optional): New conversation title
  - system_prompt (str, optional): New system prompt
  - is_archived (bool, optional): Archive/unarchive conversation

RESPONSE:
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Code review - Updated",
    "system_prompt": "You are an expert code reviewer...",
    "is_archived": false,
    "accessed_at": "2024-01-15T14:25:30Z"
  }

EXAMPLE:
  PUT /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000?user_id=550e8400-e29b-41d4-a716-446655440001&title=New%20title&is_archived=false
"""

# DELETE /api/v1/conversations/{conversation_id}
"""
Delete a conversation and all its messages.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)

RESPONSE:
  {
    "message": "Conversation deleted successfully"
  }

WARNING: This deletes all messages in the conversation permanently.

EXAMPLE:
  DELETE /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000?user_id=550e8400-e29b-41d4-a716-446655440001
"""


# ============================================================================
# MESSAGE MANAGEMENT ENDPOINTS
# ============================================================================

# GET /api/v1/conversations/{conversation_id}/messages
"""
Get messages in a conversation with pagination.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)
  - page (int, optional): Page number (1-indexed, default: 1)
  - page_size (int, optional): Messages per page (1-100, default: 50)
  - role (str, optional): Filter by role (user, assistant, system)

RESPONSE:
  {
    "total": 24,
    "page": 1,
    "page_size": 50,
    "messages": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440010",
        "role": "user",
        "content": "How do I optimize this function?",
        "tokens": 8,
        "finish_reason": null,
        "created_at": "2024-01-15T10:35:00Z"
      },
      {
        "id": "550e8400-e29b-41d4-a716-446655440011",
        "role": "assistant",
        "content": "You can optimize it by using...",
        "tokens": 156,
        "finish_reason": "stop",
        "created_at": "2024-01-15T10:35:45Z"
      },
      ...
    ]
  }

EXAMPLE (Get last 20 messages):
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/messages?user_id=550e8400-e29b-41d4-a716-446655440001&page=1&page_size=20

EXAMPLE (Get only user messages):
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/messages?user_id=550e8400-e29b-41d4-a716-446655440001&role=user
"""

# POST /api/v1/conversations/{conversation_id}/messages
"""
Add a message to a conversation.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)
  - role (str, required): Message role (user, assistant, system)
  - content (str, required): Message content/text
  - tokens (int, optional): Token count for the message

RESPONSE:
  {
    "id": "550e8400-e29b-41d4-a716-446655440012",
    "role": "user",
    "content": "What about error handling?",
    "tokens": 5,
    "created_at": "2024-01-15T10:40:00Z"
  }

NOTE: Automatically updates conversation accessed_at timestamp

EXAMPLE:
  POST /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/messages?user_id=550e8400-e29b-41d4-a716-446655440001&role=user&content=How%20do%20I%20fix%20this%20bug%3F&tokens=7
"""


# ============================================================================
# SEARCH & ANALYTICS ENDPOINTS
# ============================================================================

# GET /api/v1/conversations/{conversation_id}/search
"""
Search messages in a conversation.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)
  - query (str, required): Search query (case-insensitive substring match)

RESPONSE:
  {
    "query": "optimization",
    "total": 3,
    "messages": [
      {
        "id": "550e8400-e29b-41d4-a716-446655440011",
        "role": "assistant",
        "content": "You can optimize it by using...",
        "created_at": "2024-01-15T10:35:45Z"
      },
      ...
    ]
  }

EXAMPLE:
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/search?user_id=550e8400-e29b-41d4-a716-446655440001&query=optimization
"""

# GET /api/v1/conversations/{conversation_id}/export
"""
Export a conversation in various formats.

PATH PARAMETERS:
  - conversation_id (UUID): Conversation ID

QUERY PARAMETERS:
  - user_id (UUID, required): User ID (for authorization)
  - format (str, optional): Export format (json, markdown, default: json)

RESPONSE (format=json):
  {
    "title": "Code review session",
    "model": "deepseek-coder:6.7b",
    "system_prompt": "You are an expert code reviewer...",
    "created_at": "2024-01-15T10:30:00Z",
    "messages": [
      {
        "role": "user",
        "content": "How do I optimize this function?",
        "timestamp": "2024-01-15T10:35:00Z"
      },
      {
        "role": "assistant",
        "content": "You can optimize it by using...",
        "timestamp": "2024-01-15T10:35:45Z"
      }
    ]
  }

RESPONSE (format=markdown):
  {
    "format": "markdown",
    "content": "# Code review session\\n\\n**Model:** deepseek-coder:6.7b\\n\\n..."
  }

EXAMPLE (JSON export):
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/export?user_id=550e8400-e29b-41d4-a716-446655440001&format=json

EXAMPLE (Markdown export):
  GET /api/v1/conversations/550e8400-e29b-41d4-a716-446655440000/export?user_id=550e8400-e29b-41d4-a716-446655440001&format=markdown
"""


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
TYPICAL CONVERSATION WORKFLOW:

1. Create a conversation:
   POST /api/v1/conversations/?user_id=<uid>&model=deepseek-coder:6.7b&title=New%20chat

2. Get conversation details:
   GET /api/v1/conversations/<conv_id>?user_id=<uid>

3. Add user message:
   POST /api/v1/conversations/<conv_id>/messages?user_id=<uid>&role=user&content=<msg>&tokens=<n>

4. Add assistant response:
   POST /api/v1/conversations/<conv_id>/messages?user_id=<uid>&role=assistant&content=<response>&tokens=<n>

5. Retrieve messages:
   GET /api/v1/conversations/<conv_id>/messages?user_id=<uid>&page=1&page_size=20

6. Search conversation:
   GET /api/v1/conversations/<conv_id>/search?user_id=<uid>&query=<search_term>

7. Export conversation:
   GET /api/v1/conversations/<conv_id>/export?user_id=<uid>&format=json

8. Update conversation:
   PUT /api/v1/conversations/<conv_id>?user_id=<uid>&title=Updated&is_archived=false

9. Archive conversation:
   PUT /api/v1/conversations/<conv_id>?user_id=<uid>&is_archived=true

10. Delete conversation:
    DELETE /api/v1/conversations/<conv_id>?user_id=<uid>


CURL EXAMPLES:

Create conversation:
  curl -X POST 'http://localhost:11000/api/v1/conversations/?user_id=550e8400-e29b-41d4-a716-446655440000&model=deepseek-coder:6.7b&title=Test'

Get messages:
  curl 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001/messages?user_id=550e8400-e29b-41d4-a716-446655440000'

Add message:
  curl -X POST 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001/messages?user_id=550e8400-e29b-41d4-a716-446655440000&role=user&content=Hello'

Search:
  curl 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001/search?user_id=550e8400-e29b-41d4-a716-446655440000&query=optimization'

Export (JSON):
  curl 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001/export?user_id=550e8400-e29b-41d4-a716-446655440000&format=json'

List conversations:
  curl 'http://localhost:11000/api/v1/conversations/?user_id=550e8400-e29b-41d4-a716-446655440000&page=1&page_size=20'

Update conversation:
  curl -X PUT 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001?user_id=550e8400-e29b-41d4-a716-446655440000&title=NewTitle'

Archive conversation:
  curl -X PUT 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001?user_id=550e8400-e29b-41d4-a716-446655440000&is_archived=true'

Delete conversation:
  curl -X DELETE 'http://localhost:11000/api/v1/conversations/550e8400-e29b-41d4-a716-446655440001?user_id=550e8400-e29b-41d4-a716-446655440000'


ERROR RESPONSES:

404 Not Found:
  {
    "detail": "Conversation not found"
  }

400 Bad Request:
  {
    "detail": "Invalid role"
  }

500 Internal Server Error:
  {
    "detail": "Failed to create conversation: <error_message>"
  }
"""
