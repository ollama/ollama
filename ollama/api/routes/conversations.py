"""
Conversation History API Endpoints
Provides full conversation management and retrieval functionality.
"""

import uuid
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from ollama.models import Conversation
from ollama.repositories import RepositoryFactory, get_repositories

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversations"],
)


@router.get("")
async def list_conversations(
    user_id: uuid.UUID = Query(..., description="User ID"),
    archived: bool = Query(False, description="Include archived conversations"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """List conversations for a user.

    Args:
        user_id: User ID to list conversations for
        archived: Include archived conversations
        page: Page number for pagination
        page_size: Number of conversations per page
        repos: Repository factory dependency

    Returns:
        List of conversations with metadata
    """
    try:
        conv_repo = repos.get_conversation_repository()

        # Get paginated conversations
        conversations, total = await conv_repo.get_paginated(
            page=page,
            page_size=page_size,
            order_by="accessed_at",
            user_id=user_id,
        )

        # Filter archived if needed
        if not archived:
            conversations = [c for c in conversations if not c.is_archived]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "conversations": [
                {
                    "id": str(c.id),
                    "title": c.title,
                    "model": c.model,
                    "is_archived": c.is_archived,
                    "created_at": c.created_at.isoformat(),
                    "accessed_at": c.accessed_at.isoformat(),
                    "message_count": await repos.get_message_repository().count_conversation_messages(
                        c.id
                    ),
                }
                for c in conversations
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {e!s}") from e


@router.post("")
async def create_conversation(
    user_id: uuid.UUID = Query(..., description="User ID"),
    model: str = Query(..., description="Model to use"),
    title: str = Query(None, description="Conversation title"),
    system_prompt: str = Query(None, description="System prompt"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Create a new conversation.

    Args:
        user_id: User ID
        model: Model name to use
        title: Optional conversation title
        system_prompt: Optional system prompt
        repos: Repository factory dependency

    Returns:
        Created conversation details
    """
    try:
        conv_repo = repos.get_conversation_repository()

        conversation = await conv_repo.create_conversation(
            user_id=user_id,
            model=model,
            title=title,
            system_prompt=system_prompt,
        )

        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "model": conversation.model,
            "system_prompt": conversation.system_prompt,
            "is_archived": conversation.is_archived,
            "created_at": conversation.created_at.isoformat(),
            "accessed_at": conversation.accessed_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {e!s}") from e


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Get conversation details by ID.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        repos: Repository factory dependency

    Returns:
        Conversation details
    """
    try:
        conv_repo = repos.get_conversation_repository()

        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update accessed_at
        await conv_repo.update_accessed_at(conversation_id)

        return {
            "id": str(conversation.id),
            "title": conversation.title,
            "model": conversation.model,
            "system_prompt": conversation.system_prompt,
            "parameters": conversation.parameters,
            "is_archived": conversation.is_archived,
            "created_at": conversation.created_at.isoformat(),
            "accessed_at": conversation.accessed_at.isoformat(),
            "message_count": await repos.get_message_repository().count_conversation_messages(
                conversation_id
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {e!s}") from e


@router.put("/{conversation_id}")
async def update_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    title: str = Query(None, description="New title"),
    system_prompt: str = Query(None, description="New system prompt"),
    is_archived: bool = Query(None, description="Archive status"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Update conversation details.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        title: New title
        system_prompt: New system prompt
        is_archived: New archive status
        repos: Repository factory dependency

    Returns:
        Updated conversation details
    """
    try:
        conv_repo = repos.get_conversation_repository()

        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Update fields if provided
        if title is not None:
            await conv_repo.update_title(conversation_id, title)

        if system_prompt is not None:
            await conv_repo.update(conversation_id, system_prompt=system_prompt)

        if is_archived is not None:
            if is_archived:
                await conv_repo.archive_conversation(conversation_id)
            else:
                await conv_repo.unarchive_conversation(conversation_id)

        # Get updated conversation
        updated = cast(Conversation, await conv_repo.get_by_id(conversation_id))

        return {
            "id": str(updated.id),
            "title": updated.title,
            "system_prompt": updated.system_prompt,
            "is_archived": updated.is_archived,
            "accessed_at": updated.accessed_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {e!s}") from e


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Delete a conversation and all its messages.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        repos: Repository factory dependency

    Returns:
        Success message
    """
    try:
        conv_repo = repos.get_conversation_repository()
        msg_repo = repos.get_message_repository()

        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Delete all messages first
        await msg_repo.delete_conversation_messages(conversation_id)

        # Delete conversation
        await conv_repo.delete(conversation_id)

        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {e!s}") from e


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Messages per page"),
    role: str = Query(None, description="Filter by role (user, assistant, system)"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Get messages in a conversation.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        page: Page number
        page_size: Messages per page
        role: Optional role filter
        repos: Repository factory dependency

    Returns:
        Paginated messages
    """
    try:
        conv_repo = repos.get_conversation_repository()
        msg_repo = repos.get_message_repository()

        # Verify ownership
        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        if role:
            messages = await msg_repo.get_messages_by_role(conversation_id, role)
        else:
            messages = await msg_repo.get_by_conversation_id(conversation_id)

        # Manual pagination
        total = len(messages)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = messages[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "tokens": m.tokens,
                    "finish_reason": m.finish_reason,
                    "created_at": m.created_at.isoformat(),
                }
                for m in paginated
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {e!s}") from e


@router.post("/{conversation_id}/messages")
async def add_message(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    role: str = Query(..., description="Message role (user, assistant, system)"),
    content: str = Query(..., description="Message content"),
    tokens: int = Query(None, description="Token count"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Add a message to a conversation.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        role: Message role
        content: Message content
        tokens: Optional token count
        repos: Repository factory dependency

    Returns:
        Created message details
    """
    try:
        conv_repo = repos.get_conversation_repository()
        msg_repo = repos.get_message_repository()

        # Verify ownership
        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Add message
        if role == "user":
            message = await msg_repo.add_user_message(
                conversation_id=conversation_id,
                content=content,
                tokens=tokens,
            )
        elif role == "assistant":
            message = await msg_repo.add_assistant_message(
                conversation_id=conversation_id,
                content=content,
                tokens=tokens,
            )
        elif role == "system":
            message = await msg_repo.add_system_message(
                conversation_id=conversation_id,
                content=content,
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid role")

        # Update conversation accessed_at
        await conv_repo.update_accessed_at(conversation_id)

        return {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "tokens": message.tokens,
            "created_at": message.created_at.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {e!s}") from e


@router.get("/{conversation_id}/search")
async def search_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    query: str = Query(..., description="Search query"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Search messages in a conversation.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        query: Search query
        repos: Repository factory dependency

    Returns:
        Matching messages
    """
    try:
        conv_repo = repos.get_conversation_repository()
        msg_repo = repos.get_message_repository()

        # Verify ownership
        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Search messages
        messages = await msg_repo.search_messages(conversation_id, query)

        return {
            "query": query,
            "total": len(messages),
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search: {e!s}") from e


@router.get("/{conversation_id}/export")
async def export_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID = Query(..., description="User ID"),
    format: str = Query("json", description="Export format (json, markdown, txt)"),
    repos: RepositoryFactory = Depends(get_repositories),
) -> dict[str, Any]:
    """Export a conversation in various formats.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (for authorization)
        format: Export format
        repos: Repository factory dependency

    Returns:
        Exported conversation data
    """
    try:
        conv_repo = repos.get_conversation_repository()
        msg_repo = repos.get_message_repository()

        # Verify ownership
        conversation = await conv_repo.get_by_id(conversation_id)
        if not conversation or conversation.user_id != user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get all messages
        messages = await msg_repo.get_by_conversation_id(conversation_id)

        if format == "json":
            return {
                "title": conversation.title,
                "model": conversation.model,
                "system_prompt": conversation.system_prompt,
                "created_at": conversation.created_at.isoformat(),
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.created_at.isoformat(),
                    }
                    for m in messages
                ],
            }

        elif format == "markdown":
            markdown = f"# {conversation.title}\n\n"
            markdown += f"**Model:** {conversation.model}\n\n"
            if conversation.system_prompt:
                markdown += f"**System Prompt:** {conversation.system_prompt}\n\n"
            markdown += "## Conversation\n\n"

            for m in messages:
                markdown += f"### {m.role.upper()}\n\n{m.content}\n\n"

            return {"format": "markdown", "content": markdown}

        else:
            raise HTTPException(status_code=400, detail="Invalid export format")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export: {e!s}") from e
