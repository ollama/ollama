"""Error handlers and structured response formatting.

Provides centralized error handling and consistent response formatting
for all API endpoints.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ollama.exceptions import OllamaException


class StructuredResponse:
    """Base class for structured API responses.

    Attributes:
        success: Whether operation was successful
        data: Response data
        error: Error details if unsuccessful
        metadata: Request metadata (request_id, timestamp)
    """

    def __init__(
        self,
        success: bool,
        data: Optional[dict[str, Any]] = None,
        error: Optional[dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Initialize structured response.

        Args:
            success: Operation success status
            data: Response payload
            error: Error details
            request_id: Request identifier
        """
        self.success = success
        self.data = data
        self.error = error
        self.request_id = request_id or str(uuid4())
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary.

        Returns:
            Response dictionary with standard format
        """
        response: dict[str, Any] = {
            "success": self.success,
            "metadata": {
                "request_id": self.request_id,
                "timestamp": self.timestamp,
            },
        }

        if self.success and self.data is not None:
            response["data"] = self.data
        elif not self.success and self.error is not None:
            response["error"] = self.error

        return response


def create_success_response(
    data: dict[str, Any],
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create successful response.

    Args:
        data: Response payload
        request_id: Optional request identifier

    Returns:
        Structured success response
    """
    return StructuredResponse(
        success=True,
        data=data,
        request_id=request_id,
    ).to_dict()


def create_error_response(
    error: OllamaException,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Create error response from exception.

    Args:
        error: OllamaException instance
        request_id: Optional request identifier

    Returns:
        Structured error response
    """
    return StructuredResponse(
        success=False,
        error={
            "code": error.code,
            "message": error.message,
            "details": error.details,
        },
        request_id=request_id,
    ).to_dict()


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with FastAPI app.

    Args:
        app: FastAPI application instance
    """

    @app.exception_handler(OllamaException)
    async def ollama_exception_handler(
        request: Request,
        exc: OllamaException,
    ) -> JSONResponse:
        """Handle Ollama-specific exceptions.

        Args:
            request: HTTP request
            exc: Exception instance

        Returns:
            JSON response with error details
        """
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        exc.log()  # Log exception for monitoring

        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(exc, request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle Pydantic validation errors.

        Args:
            request: HTTP request
            exc: Validation error

        Returns:
            JSON response with validation details
        """
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Format validation errors
        errors = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"][1:])
            errors.append(
                {
                    "field": field,
                    "type": error["type"],
                    "message": error["msg"],
                }
            )

        response = StructuredResponse(
            success=False,
            error={
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": errors},
            },
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response.to_dict(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions.

        Args:
            request: HTTP request
            exc: Exception instance

        Returns:
            JSON response with generic error
        """
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Log unexpected exception
        import logging

        logger = logging.getLogger(__name__)
        logger.exception(
            f"Unhandled exception in request {request_id}",
            extra={"path": request.url.path},
        )

        response = StructuredResponse(
            success=False,
            error={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {
                    "type": exc.__class__.__name__,
                },
            },
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response.to_dict(),
        )
