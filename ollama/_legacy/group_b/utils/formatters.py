"""Formatting utilities for Ollama API responses."""

from datetime import UTC, datetime
from typing import Any


def format_response(
    data: Any,
    success: bool = True,
    metadata: dict[str, Any] | None = None,
    response_type: str | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    """Format API response in standard envelope.

    Args:
        data: Response data.
        success: Whether request was successful.
        metadata: Optional request metadata.
        response_type: Optional type of response.
        is_error: Whether this is an error response (alias for NOT success).

    Returns:
        Structured response dictionary.
    """
    # Align success and is_error
    if is_error:
        success = False

    merged_metadata = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": "req-" + str(int(datetime.now(UTC).timestamp())),
    }
    if metadata:
        merged_metadata.update(metadata)

    response = {
        "success": success,
        "data": data,
        "metadata": merged_metadata,
    }

    if response_type:
        response["metadata"]["type"] = response_type

    return response
